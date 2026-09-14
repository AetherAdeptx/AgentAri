#include "firstagent/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace firstagent::text {
namespace {

constexpr std::size_t minimum_ascii_capacity = 256U;
constexpr std::array<char, 8U> predictor_checkpoint_magic{{'F', 'A', 'C', 'K', 'P', 'T', '0', '2'}};

template <typename Value>
bool write_binary(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(output);
}

template <typename Value>
bool read_binary(std::istream& input, Value& value) {
    input.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(input);
}

bool write_float_vectors(std::ostream& output, const std::vector<std::vector<float>>& vectors) {
    const std::uint64_t count = vectors.size();
    if (!write_binary(output, count)) {
        return false;
    }
    for (const std::vector<float>& values : vectors) {
        const std::uint64_t size = values.size();
        if (!write_binary(output, size)) {
            return false;
        }
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!output) {
            return false;
        }
    }
    return true;
}

bool read_float_vectors(std::istream& input, std::vector<std::vector<float>>& vectors,
                        std::size_t maximum_vector_count = 4096U,
                        std::size_t maximum_total_elements = 64U * 1024U * 1024U) {
    std::uint64_t count = 0U;
    if (!read_binary(input, count) || count > maximum_vector_count) {
        return false;
    }
    vectors.assign(static_cast<std::size_t>(count), {});
    std::size_t total_elements = 0U;
    for (std::vector<float>& values : vectors) {
        std::uint64_t size = 0U;
        if (!read_binary(input, size) || size > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            size > maximum_total_elements ||
            static_cast<std::size_t>(size) > maximum_total_elements - total_elements) {
            return false;
        }
        total_elements += static_cast<std::size_t>(size);
        values.resize(static_cast<std::size_t>(size));
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input) {
            return false;
        }
    }
    return true;
}

std::uint64_t predictor_configuration_fingerprint(const LayeredTokenizer& tokenizer,
                                                   const PredictorConfig& config,
                                                   const nn::TransformerConfig& model_config) {
    std::uint64_t hash = tokenizer.fingerprint();
    const auto add_u64 = [&hash](std::uint64_t value) {
        hash ^= value + 0x9E37'79B9'7F4A'7C15ULL + (hash << 6U) + (hash >> 2U);
    };
    add_u64(config.maximum_context_words);
    add_u64(config.model_vocabulary_limit);
    add_u64(config.model_dimension);
    add_u64(config.layer_count);
    add_u64(config.attention_head_count);
    add_u64(config.key_value_head_count);
    add_u64(config.feed_forward_dimension);
    add_u64(config.maximum_sequence_length);
    add_u64(config.training_repeats_per_observation);
    add_u64(config.gradient_accumulation_steps);
    add_u64(config.maximum_training_windows_per_observation);
    add_u64(config.training_window_stride);
    add_u64(model_config.vocabulary_size);
    add_u64(model_config.maximum_sequence_length);
    add_u64(model_config.model_dimension);
    add_u64(model_config.layer_count);
    add_u64(model_config.attention_head_count);
    add_u64(model_config.key_value_head_count);
    add_u64(model_config.feed_forward_dimension);
    add_u64(config.tokenizer_neural.ascii_node_count);
    add_u64(config.tokenizer_neural.primitive_node_count);
    add_u64(config.tokenizer_neural.word_node_count);
    add_u64(config.tokenizer_neural.context_node_count);
    add_u64(config.tokenizer_neural.grammar_node_count);
    add_u64(config.tokenizer_neural.expert_count);
    add_u64(config.tokenizer_neural.region_count);
    add_u64(config.grammar.normalize_spacing ? 1U : 0U);
    add_u64(config.grammar.capitalize_sentences ? 1U : 0U);
    add_u64(config.grammar.append_terminal_punctuation ? 1U : 0U);
    add_u64(config.grammar.maximum_text_bytes);
    add_u64(config.grammar_postprocess ? 1U : 0U);
    return hash;
}

std::vector<std::string> split_words(std::string_view input) {
    std::vector<std::string> result;
    std::string current;
    const auto flush = [&result, &current]() {
        if (!current.empty()) {
            result.push_back(std::move(current));
            current.clear();
        }
    };

    for (const unsigned char value : input) {
        if (std::isalpha(value) != 0 || std::isdigit(value) != 0 || value == '\'') {
            current.push_back(static_cast<char>(value));
        } else {
            flush();
        }
    }
    flush();
    return result;
}

bool is_known_word(const LayeredTokenizer& tokenizer, TokenId token) {
    return tokenizer.is_known_word_token(token);
}

std::uint8_t default_ascii_depth(std::size_t value) {
    if ((value >= 65U && value <= 90U) || (value >= 97U && value <= 122U)) {
        return 224U;
    }
    if (value >= 48U && value <= 57U) {
        return 128U;
    }
    if (value >= 32U && value <= 126U) {
        return 48U;
    }
    if (value >= 128U) {
        return 160U;
    }
    return value >= 9U && value <= 13U ? 24U : 16U;
}

}  // namespace

LayeredTokenizer::LayeredTokenizer(TokenizerConfig config)
    : config_(config), unknown_word_id_(0U) {
    if (config_.ascii_base_capacity != minimum_ascii_capacity ||
        config_.ascii_expansion_capacity == 0U ||
        config_.primitive_rule_capacity == 0U ||
        config_.primitive_expansion_capacity == 0U ||
        config_.vocabulary_capacity == 0U) {
        throw std::invalid_argument("the ASCII base layer must contain exactly 256 tokens");
    }

    const std::size_t ascii_capacity =
        config_.ascii_base_capacity + config_.ascii_expansion_capacity;
    const std::size_t total_primitive_capacity = this->primitive_capacity();
    records_.reserve(ascii_capacity + total_primitive_capacity +
                     config_.vocabulary_capacity);
    for (std::size_t index = 0U; index < ascii_capacity; ++index) {
        records_.push_back(TokenRecord{
            .id = static_cast<TokenId>(index),
            .layer = TokenLayer::ascii,
            .text = index < config_.ascii_base_capacity
                        ? std::string(1U, static_cast<char>(index))
                        : "<unused-ascii>",
            .semantic_role = "byte",
            .rank = 0U,
            .depth = default_ascii_depth(index),
            .mobility = mobility_for_depth(default_ascii_depth(index)),
        });
    }

    // The unknown word token is the first word-layer slot. Primitive slots are
    // allocated before it, so the complete word layer can grow independently.
    for (std::size_t index = 0U; index < total_primitive_capacity; ++index) {
        records_.push_back(TokenRecord{
            .id = static_cast<TokenId>(ascii_capacity + index),
            .layer = TokenLayer::primitive,
            .text = "<unused-primitive>",
            .semantic_role = "unassigned",
            .rank = 0U,
            .depth = 160U,
            .mobility = mobility_for_depth(160U),
        });
    }
    unknown_word_id_ = static_cast<TokenId>(ascii_capacity + total_primitive_capacity);
    records_.push_back(TokenRecord{
        .id = unknown_word_id_,
        .layer = TokenLayer::word,
        .text = "<unk>",
        .semantic_role = "unknown_word",
        .rank = 0U,
        .depth = 96U,
        .mobility = mobility_for_depth(96U),
    });
}

const TokenizerConfig& LayeredTokenizer::config() const noexcept {
    return config_;
}

std::size_t LayeredTokenizer::primitive_count() const noexcept {
    return primitive_ids_.size();
}

std::size_t LayeredTokenizer::primitive_capacity() const noexcept {
    return config_.primitive_rule_capacity + config_.primitive_expansion_capacity;
}

std::size_t LayeredTokenizer::vocabulary_size() const noexcept {
    return word_ids_.size();
}

std::size_t LayeredTokenizer::vocabulary_capacity() const noexcept {
    return config_.vocabulary_capacity;
}

std::size_t LayeredTokenizer::remaining_vocabulary_slots() const noexcept {
    return config_.vocabulary_capacity > vocabulary_size()
               ? config_.vocabulary_capacity - vocabulary_size()
               : 0U;
}

std::size_t LayeredTokenizer::word_token_count() const noexcept {
    return word_tokens_.size();
}

std::size_t LayeredTokenizer::token_count() const noexcept {
    return records_.size();
}

std::size_t LayeredTokenizer::model_vocabulary_size() const noexcept {
    // The final dense class is the unknown-word class.
    return word_tokens_.size() + 1U;
}

std::size_t LayeredTokenizer::word_index(TokenId token) const noexcept {
    const auto found = word_indices_.find(token);
    return found == word_indices_.end() ? word_tokens_.size() : found->second;
}

TokenId LayeredTokenizer::token_from_word_index(std::size_t index) const noexcept {
    return index < word_tokens_.size() ? word_tokens_[index] : unknown_word_id_;
}

bool LayeredTokenizer::is_known_word_token(TokenId token) const noexcept {
    if (token >= records_.size()) {
        return false;
    }
    const TokenRecord& entry = records_[token];
    return entry.layer == TokenLayer::word && entry.id != unknown_word_id_;
}

float LayeredTokenizer::token_mobility(TokenId token) const noexcept {
    return token < records_.size() ? records_[token].mobility : 1.0F;
}

TokenId LayeredTokenizer::add_primitive(std::string primitive,
                                        std::string semantic_role,
                                        std::uint8_t depth) {
    const bool suffix_only = primitive.size() > 1U && primitive.front() == '-';
    const bool prefix_only = primitive.size() > 1U && primitive.back() == '-';
    if (suffix_only) {
        primitive.erase(primitive.begin());
    }
    if (prefix_only) {
        primitive.pop_back();
    }
    primitive = normalize_word(primitive);
    if (primitive.empty()) {
        throw std::invalid_argument("cannot add an empty primitive");
    }
    const std::string key = (prefix_only ? "^" : "") + primitive + (suffix_only ? "$" : "");
    if (const auto found = primitive_ids_.find(key); found != primitive_ids_.end()) {
        return found->second;
    }
    if (primitive_ids_.size() >= primitive_capacity()) {
        throw std::length_error("primitive layer capacity exhausted");
    }

    const TokenId id = static_cast<TokenId>(
        config_.ascii_base_capacity + config_.ascii_expansion_capacity + primitive_ids_.size());
    records_[id].text = primitive;
    records_[id].semantic_role = std::move(semantic_role);
    records_[id].depth = depth;
    records_[id].mobility = mobility_for_depth(depth);
    records_[id].prefix_only = prefix_only;
    records_[id].suffix_only = suffix_only;
    primitive_ids_.emplace(key, id);
    return id;
}

TokenId LayeredTokenizer::add_word(std::string word, std::size_t rank) {
    word = normalize_word(word);
    if (word.empty()) {
        throw std::invalid_argument("cannot add an empty word");
    }
    if (word == "<unk>") {
        return unknown_word_id_;
    }
    if (const auto found = word_ids_.find(word); found != word_ids_.end()) {
        return found->second;
    }
    if (word_ids_.size() >= config_.vocabulary_capacity) {
        throw std::length_error("word vocabulary capacity exhausted");
    }

    const TokenId id = static_cast<TokenId>(records_.size());
    records_.push_back(TokenRecord{
        .id = id,
        .layer = TokenLayer::word,
        .text = word,
        .semantic_role = "complete_word",
        .rank = rank,
        .depth = 200U,
        .mobility = mobility_for_depth(200U),
    });
    word_ids_.emplace(word, id);
    word_indices_.emplace(id, word_tokens_.size());
    word_tokens_.push_back(id);
    return id;
}

bool LayeredTokenizer::load_primitive_file(const std::string& path,
                                           std::size_t maximum_primitives,
                                           std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open primitive file: " + path;
        return false;
    }

    std::string line;
    std::size_t loaded = 0U;
    std::size_t line_number = 0U;
    try {
        while (loaded < maximum_primitives && std::getline(input, line)) {
            ++line_number;
            if (line.empty() || line.front() == '#') {
                continue;
            }

            std::istringstream fields(line);
            std::size_t layer_id = 0U;
            std::string primitive;
            if (!(fields >> layer_id >> primitive)) {
                error = "invalid primitive row at line " + std::to_string(line_number);
                return false;
            }
            std::string semantic_role;
            std::size_t depth = 160U;
            (void)(fields >> semantic_role >> depth);
            if (depth > std::numeric_limits<std::uint8_t>::max()) {
                error = "primitive depth is outside byte range at line " + std::to_string(line_number);
                return false;
            }
            (void)layer_id;
            const std::size_t before = primitive_count();
            (void)add_primitive(std::move(primitive), std::move(semantic_role),
                                static_cast<std::uint8_t>(depth));
            if (primitive_count() != before) {
                ++loaded;
            }
        }
    } catch (const std::exception& exception) {
        error = "failed loading primitives at line " + std::to_string(line_number) + ": " +
                exception.what();
        return false;
    }
    return true;
}

bool LayeredTokenizer::load_depth_map_file(const std::string& path, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open token depth map: " + path;
        return false;
    }
    std::string line;
    std::size_t line_number = 0U;
    try {
        while (std::getline(input, line)) {
            ++line_number;
            if (line.empty() || line.front() == '#' || line.rfind("region", 0U) == 0U) {
                continue;
            }
            std::istringstream fields(line);
            std::string region;
            std::string first;
            std::string last;
            std::size_t depth = 0U;
            float mobility = 0.0F;
            if (!(fields >> region >> first >> last >> depth >> mobility)) {
                error = "invalid token depth row at line " + std::to_string(line_number);
                return false;
            }
            if (first == "-" || last == "-") {
                continue;
            }
            const std::size_t first_byte = std::stoull(first);
            const std::size_t last_byte = std::stoull(last);
            if (first_byte > last_byte || last_byte >= minimum_ascii_capacity ||
                depth > std::numeric_limits<std::uint8_t>::max() || !(mobility >= 0.0F && mobility <= 1.0F)) {
                error = "invalid token depth range at line " + std::to_string(line_number);
                return false;
            }
            for (std::size_t byte = first_byte; byte <= last_byte; ++byte) {
                records_[byte].depth = static_cast<std::uint8_t>(depth);
                records_[byte].mobility = mobility;
                records_[byte].semantic_role = region;
            }
        }
    } catch (const std::exception& exception) {
        error = "failed loading token depth map at line " + std::to_string(line_number) + ": " +
                exception.what();
        return false;
    }
    return true;
}

bool LayeredTokenizer::load_vocabulary_file(const std::string& path,
                                             std::size_t maximum_words,
                                             std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open vocabulary file: " + path;
        return false;
    }

    std::string line;
    std::size_t loaded = 0U;
    std::size_t line_number = 0U;
    try {
        while (loaded < maximum_words && std::getline(input, line)) {
            ++line_number;
            if (line.empty() || line.front() == '#' || line == "rank\tword" || line == "rank word") {
                continue;
            }

            std::istringstream fields(line);
            std::size_t rank = 0U;
            std::string word;
            if (!(fields >> rank >> word)) {
                error = "invalid vocabulary row at line " + std::to_string(line_number);
                return false;
            }
            const std::size_t before = vocabulary_size();
            (void)add_word(std::move(word), rank);
            if (vocabulary_size() != before) {
                ++loaded;
            }
        }
    } catch (const std::exception& exception) {
        error = "failed loading vocabulary at line " + std::to_string(line_number) + ": " +
                exception.what();
        return false;
    }
    return true;
}

TokenId LayeredTokenizer::lookup_word(std::string_view word) const noexcept {
    const std::string normalized = normalize_word(word);
    const auto found = word_ids_.find(normalized);
    return found == word_ids_.end() ? unknown_word_id_ : found->second;
}

const TokenRecord& LayeredTokenizer::record(TokenId id) const {
    if (id >= records_.size()) {
        throw std::out_of_range("token id is outside the tokenizer map");
    }
    return records_[id];
}

std::string_view LayeredTokenizer::text(TokenId id) const {
    return record(id).text;
}

std::vector<TokenId> LayeredTokenizer::encode_words(std::string_view input) const {
    const std::vector<std::string> words = split_words(input);
    std::vector<TokenId> result;
    result.reserve(words.size());
    for (const std::string& word : words) {
        result.push_back(lookup_word(word));
    }
    return result;
}

std::vector<TokenId> LayeredTokenizer::encode_text(std::string_view input) const {
    std::vector<TokenId> result;
    result.reserve(input.size());
    std::string current;
    const auto flush_word = [&]() {
        if (current.empty()) {
            return;
        }
        const TokenId word = lookup_word(current);
        if (is_known_word_token(word)) {
            result.push_back(word);
        } else {
            const std::vector<TokenId> parts = encode_word_parts(current);
            result.insert(result.end(), parts.begin(), parts.end());
        }
        current.clear();
    };
    for (const unsigned char value : input) {
        if (std::isalpha(value) != 0 || std::isdigit(value) != 0 || value == '\'') {
            current.push_back(static_cast<char>(value));
        } else {
            flush_word();
            result.push_back(static_cast<TokenId>(value));
        }
    }
    flush_word();
    return result;
}

std::string LayeredTokenizer::decode_text(const std::vector<TokenId>& tokens) const {
    std::string result;
    for (const TokenId token : tokens) {
        if (token < config_.ascii_base_capacity) {
            result.push_back(static_cast<char>(token));
            continue;
        }
        const TokenRecord& entry = record(token);
        if (entry.layer == TokenLayer::primitive ||
            (entry.layer == TokenLayer::word && entry.text != "<unk>")) {
            result += entry.text;
        }
    }
    return result;
}

std::uint64_t LayeredTokenizer::fingerprint() const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto add_byte = [&hash](std::uint8_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    const auto add_u64 = [&add_byte](std::uint64_t value) {
        for (unsigned byte = 0U; byte < 8U; ++byte) {
            add_byte(static_cast<std::uint8_t>(value & 0xFFU));
            value >>= 8U;
        }
    };
    const auto add_string = [&add_byte](const std::string& value) {
        for (const unsigned char character : value) {
            add_byte(character);
        }
        add_byte(0U);
    };
    add_u64(config_.ascii_base_capacity);
    add_u64(config_.ascii_expansion_capacity);
    add_u64(config_.primitive_rule_capacity);
    add_u64(config_.primitive_expansion_capacity);
    add_u64(config_.vocabulary_capacity);
    for (const TokenRecord& entry : records_) {
        add_u64(entry.id);
        add_byte(static_cast<std::uint8_t>(entry.layer));
        add_u64(entry.rank);
        add_byte(entry.depth);
        add_byte(entry.prefix_only ? 1U : 0U);
        add_byte(entry.suffix_only ? 1U : 0U);
        add_string(entry.text);
        add_string(entry.semantic_role);
    }
    return hash;
}

std::string LayeredTokenizer::decode_words(const std::vector<TokenId>& tokens) const {
    std::ostringstream output;
    bool first = true;
    for (const TokenId token : tokens) {
        const std::string_view value = text(token);
        if (value == "<unk>" || value == "<unused-primitive>") {
            continue;
        }
        if (!first) {
            output << ' ';
        }
        output << value;
        first = false;
    }
    return output.str();
}

std::vector<TokenId> LayeredTokenizer::encode_word_parts(std::string_view word) const {
    const std::string normalized = normalize_word(word);
    if (normalized.empty()) {
        return {};
    }

    const TokenId complete_word = lookup_word(normalized);
    if (complete_word != unknown_word_id_) {
        return {complete_word};
    }

    std::vector<TokenId> result;
    std::size_t position = 0U;
    while (position < normalized.size()) {
        std::size_t best_length = 0U;
        TokenId best_token = 0U;
        for (const auto& [key, token] : primitive_ids_) {
            (void)key;
            const TokenRecord& rule = record(token);
            const std::string& primitive = rule.text;
            if (primitive.size() < 2U || primitive.size() <= best_length ||
                position + primitive.size() > normalized.size()) {
                continue;
            }
            if ((rule.prefix_only && position != 0U) ||
                (rule.suffix_only && position + primitive.size() != normalized.size())) {
                continue;
            }
            if (normalized.compare(position, primitive.size(), primitive) == 0) {
                best_length = primitive.size();
                best_token = token;
            }
        }

        if (best_length > 0U) {
            result.push_back(best_token);
            position += best_length;
        } else {
            result.push_back(static_cast<TokenId>(
                static_cast<unsigned char>(normalized[position])));
            ++position;
        }
    }
    return result;
}

std::string LayeredTokenizer::decode_word_parts(const std::vector<TokenId>& tokens) const {
    std::string result;
    for (const TokenId token : tokens) {
        const TokenRecord& entry = record(token);
        if (entry.layer == TokenLayer::ascii || entry.layer == TokenLayer::primitive) {
            if (entry.text != "<unused-primitive>") {
                result += entry.text;
            }
        } else if (entry.layer == TokenLayer::word && entry.text != "<unk>") {
            result += entry.text;
        }
    }
    return result;
}

std::vector<TokenId> LayeredTokenizer::encode_ascii(std::string_view input) {
    std::vector<TokenId> result;
    result.reserve(input.size());
    for (const unsigned char value : input) {
        result.push_back(static_cast<TokenId>(value));
    }
    return result;
}

std::string LayeredTokenizer::normalize_word(std::string_view word) {
    std::string result;
    result.reserve(word.size());
    for (const unsigned char value : word) {
        result.push_back(static_cast<char>(std::tolower(value)));
    }
    return result;
}

float LayeredTokenizer::mobility_for_depth(std::uint8_t depth) noexcept {
    return std::clamp(1.0F - static_cast<float>(depth) / 255.0F, 0.05F, 1.0F);
}

std::vector<TokenId> ContextWindow::flatten(std::size_t maximum_words) const {
    const std::size_t total = goal_words.size() + frame_words.size() +
                              sampled_history_words.size() + recent_words.size();
    std::vector<TokenId> result;
    if (maximum_words == 0U || total <= maximum_words) {
        result.reserve(total);
        result.insert(result.end(), goal_words.begin(), goal_words.end());
        result.insert(result.end(), frame_words.begin(), frame_words.end());
        result.insert(result.end(), sampled_history_words.begin(), sampled_history_words.end());
        result.insert(result.end(), recent_words.begin(), recent_words.end());
        return result;
    }

    // Keep the live suffix and goal preferentially. Historical summaries fill
    // the remaining capacity instead of evicting the current conversation.
    result.reserve(maximum_words);
    const std::size_t recent_count = std::min(recent_words.size(), maximum_words);
    const std::size_t goal_budget = maximum_words - recent_count;
    const std::size_t goal_count = std::min(goal_words.size(), goal_budget);
    result.insert(result.end(), goal_words.begin(),
                  goal_words.begin() + static_cast<std::ptrdiff_t>(goal_count));
    std::size_t history_budget = maximum_words - goal_count - recent_count;
    const auto append_limited = [&result, &history_budget](const std::vector<TokenId>& values) {
        const std::size_t count = std::min(values.size(), history_budget);
        result.insert(result.end(), values.begin(),
                      values.begin() + static_cast<std::ptrdiff_t>(count));
        history_budget -= count;
    };
    append_limited(frame_words);
    append_limited(sampled_history_words);
    result.insert(result.end(), recent_words.end() - static_cast<std::ptrdiff_t>(recent_count),
                  recent_words.end());
    return result;
}

ContextSteering::ContextSteering(const LayeredTokenizer& tokenizer,
                                 ContextSteeringConfig config)
    : tokenizer_(&tokenizer), config_(config) {
    if (config_.local_window_words == 0U || config_.history_sample_count == 0U ||
        config_.frame_word_count == 0U || config_.salient_words_per_frame == 0U ||
        config_.retained_frame_count == 0U) {
        throw std::invalid_argument("context steering capacities must be non-zero");
    }
}

void ContextSteering::record_chat(std::uint64_t log_number,
                                  const std::vector<TokenId>& words) {
    logs_[log_number] = words;
    rebuild_frames();
}

void ContextSteering::record_chat_text(std::uint64_t log_number, std::string_view text) {
    record_chat(log_number, tokenizer_->encode_words(text));
}

void ContextSteering::set_goal(std::uint64_t log_number,
                               const std::vector<TokenId>& words) {
    goals_[log_number] = words;
}

void ContextSteering::set_goal_text(std::uint64_t log_number, std::string_view text) {
    set_goal(log_number, tokenizer_->encode_words(text));
}

std::vector<TokenId> ContextSteering::flatten_history(std::uint64_t log_number) const {
    std::vector<TokenId> history;
    for (const auto& [entry_log, words] : logs_) {
        if (entry_log >= log_number) {
            break;
        }
        history.insert(history.end(), words.begin(), words.end());
    }
    return history;
}

std::vector<TokenId> ContextSteering::sample_history(const std::vector<TokenId>& history,
                                                     std::uint64_t log_number) const {
    if (history.empty()) {
        return {};
    }

    std::mt19937_64 generator(log_number ^ 0x9E37'79B9'7F4A'7C15ULL);
    std::uniform_real_distribution<double> random_unit(0.0, 1.0);
    std::vector<TokenId> result;
    result.reserve(config_.history_sample_count);
    for (std::size_t sample = 0U; sample < config_.history_sample_count; ++sample) {
        const double unit = random_unit(generator);
        const std::size_t maximum_distance = history.size() - 1U;
        const std::size_t distance = static_cast<std::size_t>(
            unit * unit * static_cast<double>(maximum_distance));
        const std::size_t slot = maximum_distance - distance;
        const std::size_t first = slot > config_.neighborhood_radius
                                       ? slot - config_.neighborhood_radius
                                       : 0U;
        const std::size_t last = std::min(history.size() - 1U,
                                           slot + config_.neighborhood_radius);

        std::size_t best = slot;
        std::size_t best_rank = 0U;
        bool found_word = false;
        for (std::size_t index = first; index <= last; ++index) {
            const TokenId token = history[index];
            if (!is_known_word(*tokenizer_, token)) {
                continue;
            }
            const std::size_t rank = tokenizer_->record(token).rank;
            if (!found_word || rank > best_rank) {
                best = index;
                best_rank = rank;
                found_word = true;
            }
        }
        result.push_back(history[best]);
    }
    return result;
}

void ContextSteering::rebuild_frames() {
    frames_.clear();
    for (const auto& [log_number, words] : logs_) {
        std::size_t frame_index = 0U;
        for (std::size_t offset = 0U; offset < words.size();
             offset += config_.frame_word_count, ++frame_index) {
            const std::size_t end = std::min(words.size(), offset + config_.frame_word_count);
            std::unordered_map<TokenId, std::size_t> counts;
            for (std::size_t index = offset; index < end; ++index) {
                if (is_known_word(*tokenizer_, words[index])) {
                    ++counts[words[index]];
                }
            }

            std::vector<TokenId> candidates;
            candidates.reserve(counts.size());
            for (const auto& [token, count] : counts) {
                (void)count;
                candidates.push_back(token);
            }
            std::sort(candidates.begin(), candidates.end(), [this, &counts](TokenId left,
                                                                              TokenId right) {
                if (counts[left] != counts[right]) {
                    return counts[left] < counts[right];
                }
                if (tokenizer_->record(left).rank != tokenizer_->record(right).rank) {
                    return tokenizer_->record(left).rank > tokenizer_->record(right).rank;
                }
                return left < right;
            });
            if (candidates.size() > config_.salient_words_per_frame) {
                candidates.resize(config_.salient_words_per_frame);
            }
            frames_.push_back(ContextFrame{
                .log_number = log_number,
                .frame_index = frame_index,
                .salient_words = std::move(candidates),
            });
        }
    }
    while (frames_.size() > config_.retained_frame_count) {
        frames_.pop_front();
    }
}

ContextWindow ContextSteering::build(std::uint64_t log_number,
                                     const std::vector<TokenId>& current_words) const {
    ContextWindow result;
    result.log_number = log_number;

    const std::size_t recent_start = current_words.size() > config_.local_window_words
                                         ? current_words.size() - config_.local_window_words
                                         : 0U;
    result.recent_words.assign(current_words.begin() + static_cast<std::ptrdiff_t>(recent_start),
                               current_words.end());

    std::vector<TokenId> history = flatten_history(log_number);
    history.insert(history.end(), current_words.begin(),
                   current_words.begin() + static_cast<std::ptrdiff_t>(recent_start));
    result.sampled_history_words = sample_history(history, log_number);

    for (const ContextFrame& frame : frames_) {
        if (frame.log_number > log_number) {
            continue;
        }
        for (const TokenId token : frame.salient_words) {
            result.frame_words.push_back(token);
        }
    }
    if (const auto goal = goals_.find(log_number); goal != goals_.end()) {
        result.goal_words = goal->second;
    }
    return result;
}

ContextWindow ContextSteering::build_text(std::uint64_t log_number,
                                          std::string_view current_text) const {
    return build(log_number, tokenizer_->encode_words(current_text));
}

std::size_t ContextSteering::frame_count() const noexcept {
    return frames_.size();
}

namespace {

bool grammar_word_byte(unsigned char value) noexcept {
    return std::isalpha(value) != 0 || std::isdigit(value) != 0 || value == '\'';
}

std::string grammar_lower(std::string_view value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char character : value) {
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    return result;
}

bool grammar_terminal(char value) noexcept {
    return value == '.' || value == '!' || value == '?';
}

bool grammar_spacing_punctuation(char value) noexcept {
    return value == '.' || value == ',' || value == '!' || value == '?' ||
           value == ';' || value == ':';
}

}  // namespace

GrammarEngine::GrammarEngine(const LayeredTokenizer& tokenizer, GrammarConfig config)
    : tokenizer_(&tokenizer), config_(config) {
    if (config_.maximum_text_bytes == 0U) {
        throw std::invalid_argument("grammar text capacity must be non-zero");
    }
}

void GrammarEngine::observe(std::string_view text) {
    const std::string_view bounded = text.substr(0U, std::min(text.size(), config_.maximum_text_bytes));
    ++state_.observations;
    state_.observed_tokens += tokenizer_->encode_words(bounded).size();
    std::string previous;
    std::string current;
    for (const unsigned char value : bounded) {
        if (grammar_word_byte(value)) {
            current.push_back(static_cast<char>(std::tolower(value)));
            continue;
        }
        if (!current.empty()) {
            if (!previous.empty()) {
                const std::string transition = previous + '\t' + current;
                auto [entry, inserted] = transitions_.try_emplace(transition, 0U);
                ++entry->second;
                if (inserted) {
                    ++state_.learned_transitions;
                }
            }
            previous = std::move(current);
            current.clear();
        }
        if (grammar_terminal(static_cast<char>(value))) {
            ++state_.observed_sentences;
            previous.clear();
        }
    }
    if (!current.empty() && !previous.empty()) {
        const std::string transition = previous + '\t' + current;
        auto [entry, inserted] = transitions_.try_emplace(transition, 0U);
        ++entry->second;
        if (inserted) {
            ++state_.learned_transitions;
        }
    }
}

void GrammarEngine::add_exception(std::string source, std::string replacement) {
    source = grammar_lower(source);
    if (source.empty() || replacement.empty()) {
        throw std::invalid_argument("grammar exceptions need source and replacement text");
    }
    exceptions_[std::move(source)] = std::move(replacement);
}

std::string GrammarEngine::repair(std::string_view text) const {
    const std::string_view bounded = text.substr(0U, std::min(text.size(), config_.maximum_text_bytes));
    std::string result;
    result.reserve(bounded.size() + 1U);
    bool pending_space = false;
    bool sentence_start = true;
    std::size_t index = 0U;
    while (index < bounded.size()) {
        const unsigned char value = static_cast<unsigned char>(bounded[index]);
        if (grammar_word_byte(value)) {
            const std::size_t first = index++;
            while (index < bounded.size() &&
                   grammar_word_byte(static_cast<unsigned char>(bounded[index]))) {
                ++index;
            }
            std::string word = grammar_lower(bounded.substr(first, index - first));
            if (const auto exception = exceptions_.find(word); exception != exceptions_.end()) {
                word = exception->second;
            }
            if (config_.normalize_spacing && pending_space && !result.empty() &&
                result.back() != ' ') {
                result.push_back(' ');
            }
            if (config_.capitalize_sentences && sentence_start) {
                for (char& character : word) {
                    const unsigned char candidate = static_cast<unsigned char>(character);
                    if (std::isalpha(candidate) != 0) {
                        character = static_cast<char>(std::toupper(candidate));
                        break;
                    }
                }
            }
            result += word;
            pending_space = false;
            sentence_start = false;
            continue;
        }

        if (std::isspace(value) != 0) {
            if (config_.normalize_spacing) {
                pending_space = !result.empty();
            } else {
                result.push_back(static_cast<char>(value));
            }
            ++index;
            continue;
        }

        const char punctuation = static_cast<char>(value);
        if (config_.normalize_spacing &&
            (grammar_spacing_punctuation(punctuation) || punctuation == ')')) {
            while (!result.empty() && result.back() == ' ') {
                result.pop_back();
            }
        }
        result.push_back(punctuation);
        pending_space = config_.normalize_spacing && grammar_spacing_punctuation(punctuation);
        if (grammar_terminal(punctuation)) {
            sentence_start = true;
        } else if (punctuation != '"' && punctuation != '\'') {
            sentence_start = false;
        }
        ++index;
    }

    if (config_.normalize_spacing) {
        while (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }
    }
    if (config_.append_terminal_punctuation && !result.empty() &&
        !grammar_terminal(result.back())) {
        result.push_back('.');
    }
    return result;
}

GrammarState GrammarEngine::state() const noexcept {
    return state_;
}

WordPredictor::WordPredictor(const LayeredTokenizer& tokenizer, PredictorConfig config)
    : tokenizer_(&tokenizer),
      config_(config),
      grammar_(tokenizer, config.grammar),
      tokenizer_neural_(tokenizer, config.tokenizer_neural),
      model_(nn::TransformerConfig{
          .vocabulary_size = std::min(tokenizer.word_token_count(), config.model_vocabulary_limit) + 1U,
          .maximum_sequence_length = config.maximum_sequence_length,
          .model_dimension = config.model_dimension,
          .layer_count = config.layer_count,
          .attention_head_count = config.attention_head_count,
          .key_value_head_count = config.key_value_head_count,
          .feed_forward_dimension = config.feed_forward_dimension,
          .rotary_dimension = 0U,
          .rope_theta = 10000.0F,
          .rms_epsilon = 1.0e-5F,
          .seed = config.seed,
      }),
      optimizer_(model_.parameters(), config.learning_rate, 0.9F, 0.999F, 1.0e-8F,
                 config.weight_decay) {
    if (config_.maximum_context_words == 0U || config_.model_vocabulary_limit == 0U ||
        config_.maximum_sequence_length == 0U || config_.training_repeats_per_observation == 0U ||
        config_.gradient_accumulation_steps == 0U ||
        config_.maximum_training_windows_per_observation == 0U) {
        throw std::invalid_argument("predictor capacities must be non-zero");
    }
}

void WordPredictor::observe(const std::vector<TokenId>& tokens) {
    tokenizer_neural_.observe_tokens(tokens);
    grammar_.observe(tokenizer_->decode_words(tokens));
    train_model_tokens(tokens);
}

void WordPredictor::train_model_tokens(const std::vector<TokenId>& tokens) {
    const std::vector<std::size_t> model_tokens = to_model_ids(tokens);
    if (model_tokens.size() < 2U) {
        return;
    }

    const std::size_t sequence_limit = std::min(config_.maximum_sequence_length, model_tokens.size());
    if (sequence_limit < 2U) {
        return;
    }
    const std::size_t final_start = model_tokens.size() > sequence_limit
                                        ? model_tokens.size() - sequence_limit
                                        : 0U;
    const std::size_t stride = config_.training_window_stride == 0U
                                   ? std::max<std::size_t>(1U, sequence_limit / 2U)
                                   : std::max<std::size_t>(1U, config_.training_window_stride);
    std::vector<std::size_t> starts;
    starts.reserve(config_.maximum_training_windows_per_observation);
    for (std::size_t start = 0U;
         start < final_start && starts.size() + 1U < config_.maximum_training_windows_per_observation;) {
        starts.push_back(start);
        if (stride > final_start - start) {
            break;
        }
        start += stride;
    }
    if (starts.empty() || starts.back() != final_start) {
        starts.push_back(final_start);
    }

    float loss_sum = 0.0F;
    for (std::size_t repeat = 0U; repeat < config_.training_repeats_per_observation; ++repeat) {
        for (std::size_t group_start = 0U; group_start < starts.size();
             group_start += config_.gradient_accumulation_steps) {
            const std::size_t group_end = std::min(
                starts.size(), group_start + config_.gradient_accumulation_steps);
            const std::size_t group_size = group_end - group_start;
            loss_sum = 0.0F;
            for (std::size_t group_index = group_start; group_index < group_end; ++group_index) {
                const std::size_t start = starts[group_index];
                const std::size_t end = std::min(model_tokens.size(), start + sequence_limit);
                std::vector<std::size_t> sequence(model_tokens.begin() +
                                                       static_cast<std::ptrdiff_t>(start),
                                                   model_tokens.begin() +
                                                       static_cast<std::ptrdiff_t>(end));
                if (sequence.size() < 2U) {
                    continue;
                }
                const std::vector<std::size_t> input(sequence.begin(), sequence.end() - 1);
                const std::vector<std::size_t> targets(sequence.begin() + 1, sequence.end());
                const nn::Tensor raw_loss = model_.loss(input, targets);
                const float raw_value = raw_loss.values().front();
                loss_sum += raw_value;
                const nn::Tensor training_loss = group_size == 1U
                                                     ? raw_loss
                                                     : raw_loss / static_cast<float>(group_size);
                training_loss.backward();
            }
            last_loss_ = loss_sum / static_cast<float>(group_size);
            optimizer_.step();
            optimizer_.zero_grad();
            tokenizer_neural_.apply_prediction_error(std::min(last_loss_ / 10.0F, 1.0F));
            ++training_steps_;
        }
    }
}

void WordPredictor::observe_text(std::string_view input) {
    tokenizer_neural_.observe_text(input);
    grammar_.observe(input);
    train_model_tokens(tokenizer_->encode_words(input));
}

bool WordPredictor::observe_file(const std::string& path,
                                 std::size_t maximum_bytes,
                                 std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "could not open training text: " + path;
        return false;
    }

    std::string line;
    std::vector<TokenId> token_batch;
    const std::size_t batch_product =
        config_.maximum_sequence_length <=
                std::numeric_limits<std::size_t>::max() /
                    config_.maximum_training_windows_per_observation
            ? config_.maximum_sequence_length * config_.maximum_training_windows_per_observation
            : std::numeric_limits<std::size_t>::max();
    const std::size_t batch_limit = std::max<std::size_t>(2U, batch_product);
    std::size_t bytes_read = 0U;
    while (std::getline(input, line)) {
        const std::size_t line_bytes = line.size() + 1U;
        const bool exceeds_limit = maximum_bytes != 0U &&
                                    (bytes_read > maximum_bytes ||
                                     line_bytes > maximum_bytes - bytes_read);
        if (exceeds_limit) {
            const std::size_t remaining = maximum_bytes - bytes_read;
            if (remaining > 0U) {
                const std::vector<TokenId> partial = tokenizer_->encode_words(
                    std::string_view(line).substr(0U, remaining));
                token_batch.insert(token_batch.end(), partial.begin(), partial.end());
            }
            break;
        }
        const std::vector<TokenId> tokens = tokenizer_->encode_words(line);
        token_batch.insert(token_batch.end(), tokens.begin(), tokens.end());
        if (token_batch.size() >= batch_limit) {
            observe(token_batch);
            token_batch.clear();
        }
        bytes_read += line_bytes;
    }
    if (!input.eof() && input.fail()) {
        error = "failed reading training text: " + path;
        return false;
    }
    if (!token_batch.empty()) {
        observe(token_batch);
    }
    return true;
}

std::vector<std::size_t> WordPredictor::to_model_ids(const std::vector<TokenId>& tokens) const {
    std::vector<std::size_t> result;
    result.reserve(tokens.size());
    const std::size_t active_words = model_.config().vocabulary_size - 1U;
    for (const TokenId token : tokens) {
        const std::size_t index = tokenizer_->word_index(token);
        result.push_back(index < active_words ? index : active_words);
    }
    return result;
}

std::vector<TokenId> WordPredictor::trim_context(const std::vector<TokenId>& tokens) const {
    const std::size_t context_limit = std::min(config_.maximum_context_words,
                                               config_.maximum_sequence_length);
    const std::size_t first = tokens.size() > context_limit
                                  ? tokens.size() - context_limit
                                  : 0U;
    return std::vector<TokenId>(tokens.begin() + static_cast<std::ptrdiff_t>(first), tokens.end());
}

std::vector<Prediction> WordPredictor::predict(const std::vector<TokenId>& context,
                                               std::size_t top_k) const {
    if (top_k == 0U) {
        return {};
    }
    const std::vector<TokenId> trimmed = trim_context(context);
    if (trimmed.empty()) {
        return {};
    }

    tokenizer_neural_.observe_tokens(trimmed);
    const std::vector<std::size_t> model_context = to_model_ids(trimmed);
    nn::NoGradGuard no_grad;
    const nn::Tensor logits = model_.forward(model_context);
    const std::size_t vocabulary = model_.config().vocabulary_size - 1U;
    const std::size_t row = (logits.shape()[0] - 1U) * logits.shape()[1];
    std::vector<std::size_t> indices(vocabulary, 0U);
    for (std::size_t index = 0U; index < vocabulary; ++index) {
        indices[index] = index;
    }
    const auto comparator = [&logits, row](std::size_t left, std::size_t right) {
        return logits.values()[row + left] > logits.values()[row + right];
    };
    const std::size_t result_count = std::min(top_k, indices.size());
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(result_count),
                      indices.end(), comparator);
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0U; index < result_count; ++index) {
        maximum = std::max(maximum, logits.values()[row + indices[index]]);
    }
    double denominator = 0.0;
    for (std::size_t index = 0U; index < result_count; ++index) {
        denominator += std::exp(static_cast<double>(logits.values()[row + indices[index]] - maximum));
    }
    std::vector<Prediction> result;
    result.reserve(result_count);
    for (std::size_t index = 0U; index < result_count; ++index) {
        const std::size_t model_index = indices[index];
        const TokenId token = tokenizer_->token_from_word_index(model_index);
        const float probability = static_cast<float>(
            std::exp(static_cast<double>(logits.values()[row + model_index] - maximum)) / denominator);
        result.push_back(Prediction{
            .token = token,
            .word = std::string(tokenizer_->text(token)),
            .count = 0U,
            .probability = probability,
        });
    }
    return result;
}

std::vector<Prediction> WordPredictor::predict_text(std::string_view context,
                                                    std::size_t top_k) const {
    return predict(tokenizer_->encode_words(context), top_k);
}

std::vector<Prediction> WordPredictor::predict(const ContextWindow& context,
                                               std::size_t top_k) const {
    return predict(context.flatten(maximum_context_words()), top_k);
}

std::string WordPredictor::generate(std::string_view prompt,
                                    std::size_t maximum_new_words,
                                    std::size_t top_k) const {
    std::vector<TokenId> tokens = tokenizer_->encode_words(prompt);
    for (std::size_t index = 0U; index < maximum_new_words; ++index) {
        const std::vector<Prediction> predictions = predict(tokens, top_k);
        if (predictions.empty()) {
            break;
        }
        tokens.push_back(predictions.front().token);
    }
    const std::string decoded = tokenizer_->decode_words(tokens);
    return config_.grammar_postprocess ? grammar_.repair(decoded) : decoded;
}

float WordPredictor::evaluate_loss(std::string_view text) const {
    const std::vector<TokenId> tokens = trim_context(tokenizer_->encode_words(text));
    if (tokens.size() < 2U) {
        return 0.0F;
    }
    const std::vector<std::size_t> model_tokens = to_model_ids(tokens);
    const std::vector<std::size_t> input(model_tokens.begin(), model_tokens.end() - 1);
    const std::vector<std::size_t> targets(model_tokens.begin() + 1, model_tokens.end());
    nn::NoGradGuard no_grad;
    return model_.loss(input, targets).values().front();
}

std::size_t WordPredictor::training_steps() const noexcept {
    return training_steps_;
}

float WordPredictor::last_loss() const noexcept {
    return last_loss_;
}

std::size_t WordPredictor::active_vocabulary_size() const noexcept {
    return model_.config().vocabulary_size - 1U;
}

TokenizerNeuralState WordPredictor::tokenizer_state() const {
    return tokenizer_neural_.state();
}

GrammarState WordPredictor::grammar_state() const noexcept {
    return grammar_.state();
}

void WordPredictor::apply_message_feedback(const std::vector<float>& feedback) {
    tokenizer_neural_.apply_message_feedback(feedback);
}

void WordPredictor::apply_prediction_error(float normalized_error) {
    tokenizer_neural_.apply_prediction_error(normalized_error);
}

bool WordPredictor::save_checkpoint(const std::string& path, std::string& error) {
    const std::filesystem::path target(path);
    const std::filesystem::path temporary = target.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "could not open predictor checkpoint for writing: " + path;
        return false;
    }
    output.write(predictor_checkpoint_magic.data(),
                 static_cast<std::streamsize>(predictor_checkpoint_magic.size()));
    const std::uint64_t active_vocabulary = active_vocabulary_size();
    const std::uint64_t parameter_count = model_.parameters().size();
    const std::uint64_t tokenizer_fingerprint = tokenizer_->fingerprint();
    const std::uint64_t configuration_fingerprint = predictor_configuration_fingerprint(
        *tokenizer_, config_, model_.config());
    if (!output || !write_binary(output, active_vocabulary) || !write_binary(output, parameter_count) ||
        !write_binary(output, tokenizer_fingerprint) ||
        !write_binary(output, configuration_fingerprint)) {
        error = "could not write predictor checkpoint header";
        return false;
    }
    for (nn::Tensor* parameter : model_.parameters()) {
        const std::uint64_t value_count = parameter->numel();
        if (!write_binary(output, value_count)) {
            error = "could not write predictor parameter size";
            return false;
        }
        output.write(reinterpret_cast<const char*>(parameter->values().data()),
                     static_cast<std::streamsize>(parameter->values().size() * sizeof(float)));
        if (!output) {
            error = "could not write predictor parameters";
            return false;
        }
    }
    const nn::AdamW::State optimizer_state = optimizer_.state();
    if (!write_binary(output, static_cast<std::uint64_t>(optimizer_state.update_count)) ||
        !write_float_vectors(output, optimizer_state.first_moment) ||
        !write_float_vectors(output, optimizer_state.second_moment) ||
        !write_binary(output, static_cast<std::uint64_t>(training_steps_)) ||
        !write_binary(output, last_loss_) || !tokenizer_neural_.save(output, error)) {
        if (error.empty()) {
            error = "could not write predictor checkpoint";
        }
        return false;
    }
    output.close();
    if (!output) {
        error = "could not finalize predictor checkpoint";
        return false;
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary, target, rename_error);
    if (rename_error) {
        std::filesystem::remove(target, rename_error);
        rename_error.clear();
        std::filesystem::rename(temporary, target, rename_error);
    }
    if (rename_error) {
        std::filesystem::remove(temporary, rename_error);
        error = "could not atomically install predictor checkpoint: " + rename_error.message();
        return false;
    }
    return true;
}

bool WordPredictor::load_checkpoint(const std::string& path, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open predictor checkpoint: " + path;
        return false;
    }
    std::array<char, predictor_checkpoint_magic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    std::uint64_t active_vocabulary = 0U;
    std::uint64_t parameter_count = 0U;
    std::uint64_t tokenizer_fingerprint = 0U;
    std::uint64_t configuration_fingerprint = 0U;
    const std::vector<nn::Tensor*> parameters = model_.parameters();
    if (!input || magic != predictor_checkpoint_magic || !read_binary(input, active_vocabulary) ||
        !read_binary(input, parameter_count) || !read_binary(input, tokenizer_fingerprint) ||
        !read_binary(input, configuration_fingerprint) ||
        active_vocabulary != active_vocabulary_size() || parameter_count != parameters.size() ||
        tokenizer_fingerprint != tokenizer_->fingerprint() ||
        configuration_fingerprint != predictor_configuration_fingerprint(
            *tokenizer_, config_, model_.config())) {
        error = "predictor checkpoint is incompatible with the active tokenizer/model configuration";
        return false;
    }
    std::vector<std::vector<float>> parameter_values;
    parameter_values.reserve(parameters.size());
    for (const nn::Tensor* parameter : parameters) {
        std::uint64_t value_count = 0U;
        if (!read_binary(input, value_count) || value_count != parameter->numel()) {
            error = "predictor checkpoint parameter shape mismatch";
            return false;
        }
        std::vector<float> values(parameter->numel(), 0.0F);
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!input) {
            error = "could not read predictor parameters";
            return false;
        }
        parameter_values.push_back(std::move(values));
    }
    std::uint64_t optimizer_updates = 0U;
    nn::AdamW::State optimizer_state;
    std::uint64_t training_steps = 0U;
    float loss = 0.0F;
    if (!read_binary(input, optimizer_updates) ||
        !read_float_vectors(input, optimizer_state.first_moment) ||
        !read_float_vectors(input, optimizer_state.second_moment) ||
        !read_binary(input, training_steps) || !read_binary(input, loss)) {
        error = "could not read predictor optimizer state";
        return false;
    }
    if (optimizer_updates > std::numeric_limits<std::size_t>::max() ||
        training_steps > std::numeric_limits<std::size_t>::max()) {
        error = "predictor checkpoint counters exceed this platform";
        return false;
    }
    optimizer_state.update_count = static_cast<std::size_t>(optimizer_updates);
    try {
        optimizer_.load_state(std::move(optimizer_state));
    } catch (const std::exception& exception) {
        error = "predictor checkpoint optimizer mismatch: " + std::string(exception.what());
        return false;
    }
    if (!tokenizer_neural_.load(input, error)) {
        return false;
    }
    for (std::size_t index = 0U; index < parameters.size(); ++index) {
        parameters[index]->mutable_values() = std::move(parameter_values[index]);
        parameters[index]->zero_grad();
    }
    training_steps_ = static_cast<std::size_t>(training_steps);
    last_loss_ = loss;
    return true;
}

std::size_t WordPredictor::maximum_context_words() const noexcept {
    return std::min(config_.maximum_context_words, config_.maximum_sequence_length);
}

}  // namespace firstagent::text
