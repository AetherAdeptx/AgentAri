#include "firstagent/TokenizerNeural.hpp"

#include "firstagent/BitPacking.hpp"
#include "firstagent/Parallel.hpp"
#include "firstagent/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <istream>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace firstagent::text {
namespace {

using FeatureVector = std::array<float, tokenizer_feature_count>;

constexpr std::uint64_t primitive_bit(PrimitiveFlag flag) {
    return std::uint64_t{1U} << static_cast<std::uint8_t>(flag);
}

float clamp_unit(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

std::uint64_t mix_token(std::uint64_t value) {
    value += 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

FeatureVector token_features(TokenId token, std::size_t layer) {
    const std::uint64_t seed = mix_token(static_cast<std::uint64_t>(token) +
                                         static_cast<std::uint64_t>(layer) * 0x10001ULL);
    FeatureVector result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
        const std::uint64_t bits = mix_token(seed + index * 0x9E37ULL);
        const float unit = static_cast<float>(bits & 0xFFFFU) / 65535.0F;
        result[index] = unit * 2.0F - 1.0F;
    }
    return result;
}

void add_scaled(FeatureVector& destination, const FeatureVector& source, float scale) {
    for (std::size_t index = 0U; index < destination.size(); ++index) {
        destination[index] += source[index] * scale;
    }
}

FeatureVector scaled(const FeatureVector& value, float factor) {
    FeatureVector result{};
    add_scaled(result, value, factor);
    return result;
}

float dot(const FeatureVector& left, const FeatureVector& right) {
    return std::inner_product(left.begin(), left.end(), right.begin(), 0.0F);
}

template <typename Value>
bool write_value(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(output);
}

template <typename Value>
bool read_value(std::istream& input, Value& value) {
    input.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(input);
}

bool write_features(std::ostream& output, const FeatureVector& features) {
    output.write(reinterpret_cast<const char*>(features.data()),
                 static_cast<std::streamsize>(features.size() * sizeof(float)));
    return static_cast<bool>(output);
}

bool read_features(std::istream& input, FeatureVector& features) {
    input.read(reinterpret_cast<char*>(features.data()),
               static_cast<std::streamsize>(features.size() * sizeof(float)));
    return static_cast<bool>(input);
}

struct LearningNode {
    FeatureVector slow_prototype{};
    FeatureVector fast_prototype{};
    float activation{0.0F};
    float trace{0.0F};
    float confidence{0.0F};
};

struct LayerUpdate {
    LayerPrimitiveSummary summary;
    FeatureVector features{};
};

class LearningNodeBank {
public:
    LearningNodeBank(std::size_t node_count,
                     std::size_t region_count,
                     std::size_t expert_count,
                     float learning_factor,
                     float activation_decay,
                     float trace_decay)
        : nodes_(node_count),
          region_count_(region_count),
          expert_count_(expert_count),
          expert_prototypes_(expert_count),
          expert_activity_(expert_count, 0.0F),
          learning_factor_(learning_factor),
          activation_decay_(activation_decay),
          trace_decay_(trace_decay) {
        if (node_count == 0U || region_count == 0U || expert_count == 0U ||
            expert_count > node_count) {
            throw std::invalid_argument("learning-node banks must be non-empty");
        }
        for (std::size_t expert = 0U; expert < expert_count_; ++expert) {
            expert_prototypes_[expert] = token_features(
                static_cast<TokenId>(expert + 1U), node_count + expert + 1U);
        }
    }

    LayerUpdate update(const LayeredTokenizer& tokenizer,
                       const std::vector<TokenId>& tokens,
                       const FeatureVector& lower_signal,
                       const FeatureVector& downward_feedback,
                       const FeatureVector& global_state,
                       float error_signal) {
        parallel::Config decay_config;
        decay_config.minimum_parallel_work = 1024U;
        parallel::parallel_for(0U, nodes_.size(), [&](std::size_t index) {
            LearningNode& node = nodes_[index];
            node.activation *= activation_decay_;
            node.trace *= trace_decay_;
        }, decay_config);
        std::fill(expert_activity_.begin(), expert_activity_.end(), 0.0F);

        std::size_t novel_hits = 0U;
        FeatureVector input_signal = scaled(global_state, 0.25F);
        add_scaled(input_signal, lower_signal, 0.50F);
        add_scaled(input_signal, downward_feedback, 0.25F);
        for (const TokenId token : tokens) {
            const float mobility = tokenizer.token_mobility(token);
            const FeatureVector features = token_features(token, nodes_.size());
            std::size_t expert = 0U;
            float expert_score = -std::numeric_limits<float>::infinity();
            for (std::size_t candidate = 0U; candidate < expert_count_; ++candidate) {
                const float tie_break = static_cast<float>(mix_token(
                    static_cast<std::uint64_t>(token) + candidate * 0x9E37ULL) & 0xFFU) /
                                        255.0F * 1.0e-3F;
                const float score = dot(expert_prototypes_[candidate], features) + tie_break;
                if (score > expert_score) {
                    expert = candidate;
                    expert_score = score;
                }
            }
            const std::size_t expert_begin = expert * nodes_.size() / expert_count_;
            const std::size_t expert_end = (expert + 1U) * nodes_.size() / expert_count_;
            const std::size_t expert_width = std::max<std::size_t>(1U, expert_end - expert_begin);
            const std::size_t node_index = expert_begin + static_cast<std::size_t>(
                mix_token(static_cast<std::uint64_t>(token)) % expert_width);
            LearningNode& node = nodes_[node_index];
            const float slow_similarity = dot(node.slow_prototype, features);
            const float fast_similarity = dot(node.fast_prototype, features);
            const float similarity = 0.5F + 0.5F *
                (0.7F * slow_similarity + 0.3F * fast_similarity) /
                static_cast<float>(features.size());
            const float feedback = 0.10F * downward_feedback[0U] + 0.05F * error_signal;
            const float activation = clamp_unit(0.55F + 0.35F * similarity + feedback);
            if (node.confidence < 0.35F) {
                ++novel_hits;
            }
            node.activation = std::max(node.activation, activation);
            expert_activity_[expert] += activation;
            node.trace = clamp_unit(node.trace + 0.25F * activation);
            node.confidence = clamp_unit(node.confidence + learning_factor_ *
                                         (activation - node.confidence));
            const float update_factor = learning_factor_ * activation * mobility;
            for (std::size_t index = 0U; index < node.slow_prototype.size(); ++index) {
                const float target = 0.75F * features[index] + 0.25F * input_signal[index];
                // Fast traces adapt immediately; slow prototypes preserve the
                // longer-term semantic direction.
                node.fast_prototype[index] += update_factor * (target - node.fast_prototype[index]);
                node.slow_prototype[index] += 0.10F * update_factor *
                                               (target - node.slow_prototype[index]);
                expert_prototypes_[expert][index] += 0.10F * update_factor *
                                                     (features[index] - expert_prototypes_[expert][index]);
                input_signal[index] += (0.7F * node.slow_prototype[index] +
                                        0.3F * node.fast_prototype[index]) * 0.005F;
            }
        }

        LayerUpdate result;
        result.summary.node_count = nodes_.size();
        result.summary.region_activations.assign(region_count_, 0.0F);
        result.summary.expert_activations = expert_activity_;
        std::vector<float> activation_values(nodes_.size(), 0.0F);
        std::vector<float> confidence_values(nodes_.size(), 0.0F);
        parallel::Config summary_config;
        summary_config.minimum_parallel_work = 1024U;
        parallel::parallel_for(0U, nodes_.size(), [&](std::size_t index) {
            const LearningNode& node = nodes_[index];
            activation_values[index] = std::max(0.0F, node.activation);
            confidence_values[index] = node.confidence;
        }, summary_config);
        float total = 0.0F;
        float weighted_position = 0.0F;
        float confidence = 0.0F;
        float entropy = 0.0F;
        for (std::size_t index = 0U; index < nodes_.size(); ++index) {
            const float activation = activation_values[index];
            total += activation;
            weighted_position += activation * static_cast<float>(index);
            confidence += confidence_values[index];
            const std::size_t region = index * region_count_ / nodes_.size();
            result.summary.region_activations[region] += activation;
        }
        result.summary.total_activation = total;
        result.summary.mean_activation = total / static_cast<float>(nodes_.size());
        result.summary.center_of_mass = total > 0.0F
                                            ? weighted_position / total /
                                                  static_cast<float>(std::max<std::size_t>(1U, nodes_.size() - 1U))
                                            : 0.0F;
        const float total_probability = std::max(total, std::numeric_limits<float>::min());
        for (const float activation : result.summary.region_activations) {
            if (activation > 0.0F) {
                const float probability = activation / total_probability;
                entropy -= probability * std::log(probability);
            }
        }
        result.summary.entropy = entropy;
        result.summary.novelty = tokens.empty()
                                     ? 0.0F
                                     : static_cast<float>(novel_hits) / static_cast<float>(tokens.size());
        result.summary.uncertainty = 1.0F - confidence / static_cast<float>(nodes_.size());
        result.summary.active_expert = static_cast<std::size_t>(std::distance(
            expert_activity_.begin(), std::max_element(expert_activity_.begin(), expert_activity_.end())));
        result.summary.upward_signal = clamp_unit(result.summary.mean_activation +
                                                  0.25F * result.summary.novelty);
        result.summary.downward_feedback = clamp_unit(std::abs(downward_feedback[0U]) +
                                                      0.25F * error_signal);
        result.features[0U] = result.summary.mean_activation;
        result.features[1U] = result.summary.center_of_mass;
        result.features[2U] = result.summary.entropy;
        result.features[3U] = result.summary.novelty;
        result.features[4U] = result.summary.uncertainty;
        result.features[5U] = result.summary.upward_signal;
        result.features[6U] = result.summary.downward_feedback;
        result.features[7U] = error_signal;
        for (std::size_t index = 8U; index < result.features.size(); ++index) {
            result.features[index] = input_signal[index];
        }
        result.summary.features = result.features;
        return result;
    }

    bool save(std::ostream& output) const {
        const std::uint64_t node_count = nodes_.size();
        const std::uint64_t expert_count = expert_count_;
        if (!write_value(output, node_count) || !write_value(output, expert_count)) {
            return false;
        }
        for (const LearningNode& node : nodes_) {
            if (!write_features(output, node.slow_prototype) || !write_features(output, node.fast_prototype) ||
                !write_value(output, node.activation) || !write_value(output, node.trace) ||
                !write_value(output, node.confidence)) {
                return false;
            }
        }
        for (const FeatureVector& prototype : expert_prototypes_) {
            if (!write_features(output, prototype)) {
                return false;
            }
        }
        output.write(reinterpret_cast<const char*>(expert_activity_.data()),
                     static_cast<std::streamsize>(expert_activity_.size() * sizeof(float)));
        return static_cast<bool>(output);
    }

    bool load(std::istream& input) {
        std::uint64_t node_count = 0U;
        std::uint64_t expert_count = 0U;
        if (!read_value(input, node_count) || !read_value(input, expert_count) ||
            node_count != nodes_.size() || expert_count != expert_count_) {
            return false;
        }
        for (LearningNode& node : nodes_) {
            if (!read_features(input, node.slow_prototype) || !read_features(input, node.fast_prototype) ||
                !read_value(input, node.activation) || !read_value(input, node.trace) ||
                !read_value(input, node.confidence)) {
                return false;
            }
        }
        for (FeatureVector& prototype : expert_prototypes_) {
            if (!read_features(input, prototype)) {
                return false;
            }
        }
        input.read(reinterpret_cast<char*>(expert_activity_.data()),
                   static_cast<std::streamsize>(expert_activity_.size() * sizeof(float)));
        return static_cast<bool>(input);
    }

private:
    std::vector<LearningNode> nodes_;
    std::size_t region_count_;
    std::size_t expert_count_;
    std::vector<FeatureVector> expert_prototypes_;
    std::vector<float> expert_activity_;
    float learning_factor_;
    float activation_decay_;
    float trace_decay_;
};

struct ParsedText {
    std::vector<TokenId> words;
    std::vector<TokenId> primitives;
    std::vector<TokenId> ascii;
    std::vector<TokenId> grammar;
    bool has_symbol{false};
    bool has_unknown_word{false};
};

ParsedText parse_text(const LayeredTokenizer& tokenizer, std::string_view input) {
    ParsedText result;
    std::string current;
    const auto flush = [&]() {
        if (current.empty()) {
            return;
        }
        const TokenId word = tokenizer.lookup_word(current);
        result.words.push_back(word);
        result.grammar.push_back(word);
        if (tokenizer.is_known_word_token(word)) {
            // Known words normally stay shallow.
            current.clear();
            return;
        }
        result.has_unknown_word = true;
        const std::vector<TokenId> parts = tokenizer.encode_word_parts(current);
        for (const TokenId part : parts) {
            const TokenRecord& record = tokenizer.record(part);
            if (record.layer == TokenLayer::primitive) {
                result.primitives.push_back(part);
            } else if (record.layer == TokenLayer::ascii) {
                result.ascii.push_back(part);
            }
        }
        current.clear();
    };
    for (const unsigned char value : input) {
        if (std::isalpha(value) != 0 || std::isdigit(value) != 0 || value == '\'') {
            current.push_back(static_cast<char>(value));
        } else {
            flush();
            if (value != ' ' && value != '\t' && value != '\n' && value != '\r') {
                result.ascii.push_back(static_cast<TokenId>(value));
                result.grammar.push_back(static_cast<TokenId>(value));
                result.has_symbol = true;
            }
        }
    }
    flush();
    return result;
}

}  // namespace

struct TokenizerNeuralStack::Impl {
    std::array<LearningNodeBank, tokenizer_layer_count> banks;
    TokenizerNeuralState state;
    FeatureVector prediction_error{};

    explicit Impl(const TokenizerNeuralConfig& config)
        : banks{
              LearningNodeBank(config.ascii_node_count, config.region_count, config.expert_count,
                               config.ascii_learning_factor, config.activation_decay, config.trace_decay),
              LearningNodeBank(config.primitive_node_count, config.region_count, config.expert_count,
                               config.primitive_learning_factor, config.activation_decay, config.trace_decay),
              LearningNodeBank(config.word_node_count, config.region_count, config.expert_count,
                               config.word_learning_factor, config.activation_decay, config.trace_decay),
              LearningNodeBank(config.context_node_count, config.region_count, config.expert_count,
                               config.context_learning_factor, config.activation_decay, config.trace_decay),
              LearningNodeBank(config.grammar_node_count, config.region_count, config.expert_count,
                               config.grammar_learning_factor, config.activation_decay, config.trace_decay),
          } {
        for (LayerPrimitiveSummary& summary : state.layers) {
            summary.region_activations.assign(config.region_count, 0.0F);
            summary.expert_activations.assign(config.expert_count, 0.0F);
        }
    }
};

TokenizerNeuralStack::TokenizerNeuralStack(const LayeredTokenizer& tokenizer,
                                           TokenizerNeuralConfig config)
    : tokenizer_(&tokenizer), config_(config), implementation_() {
    if (config_.region_count == 0U || config_.region_count > tokenizer_region_count ||
        config_.expert_count == 0U ||
        !(config_.ascii_learning_factor >= 0.0F && config_.ascii_learning_factor <= 1.0F) ||
        !(config_.primitive_learning_factor >= 0.0F && config_.primitive_learning_factor <= 1.0F) ||
        !(config_.word_learning_factor >= 0.0F && config_.word_learning_factor <= 1.0F) ||
        !(config_.context_learning_factor >= 0.0F && config_.context_learning_factor <= 1.0F) ||
        !(config_.grammar_learning_factor >= 0.0F && config_.grammar_learning_factor <= 1.0F) ||
        !(config_.activation_decay >= 0.0F && config_.activation_decay < 1.0F) ||
        !(config_.trace_decay >= 0.0F && config_.trace_decay < 1.0F) ||
        !(config_.feedback_decay >= 0.0F && config_.feedback_decay < 1.0F)) {
        throw std::invalid_argument("tokenizer neural configuration is invalid");
    }
    implementation_ = std::make_unique<Impl>(config_);
}

TokenizerNeuralStack::~TokenizerNeuralStack() = default;

void TokenizerNeuralStack::observe_text(std::string_view input) {
    const ParsedText parsed = parse_text(*tokenizer_, input);
    observe(parsed.words, parsed.primitives, parsed.ascii, parsed.words, parsed.grammar);
    if (parsed.has_symbol) {
        implementation_->state.primitives.flags |= primitive_bit(PrimitiveFlag::symbol_input);
    }
    if (parsed.has_unknown_word) {
        implementation_->state.primitives.flags |= primitive_bit(PrimitiveFlag::unknown_word);
    }
}

void TokenizerNeuralStack::observe_tokens(const std::vector<TokenId>& tokens) {
    std::vector<TokenId> words;
    std::vector<TokenId> primitives;
    std::vector<TokenId> ascii;
    bool unknown = false;
    bool symbols = false;
    for (const TokenId token : tokens) {
        if (token >= tokenizer_->config().ascii_base_capacity +
                        tokenizer_->config().ascii_expansion_capacity &&
            token < tokenizer_->token_count()) {
            const TokenRecord& entry = tokenizer_->record(token);
            if (entry.layer == TokenLayer::word) {
                words.push_back(token);
                unknown = unknown || !tokenizer_->is_known_word_token(token);
            } else if (entry.layer == TokenLayer::primitive) {
                primitives.push_back(token);
            } else {
                ascii.push_back(token);
                symbols = true;
            }
        } else {
            ascii.push_back(token);
            symbols = true;
        }
    }
    observe(words, primitives, ascii, words);
    if (unknown) {
        implementation_->state.primitives.flags |= primitive_bit(PrimitiveFlag::unknown_word);
    }
    if (symbols) {
        implementation_->state.primitives.flags |= primitive_bit(PrimitiveFlag::symbol_input);
    }
}

void TokenizerNeuralStack::observe(const std::vector<TokenId>& word_tokens,
                                   const std::vector<TokenId>& primitive_tokens,
                                   const std::vector<TokenId>& ascii_tokens,
                                   const std::vector<TokenId>& context_tokens,
                                   const std::vector<TokenId>& grammar_tokens) {
    Impl& implementation = *implementation_;
    ++implementation.state.step;
    implementation.state.primitives.flags = 0U;
    if (!ascii_tokens.empty()) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::ascii_input);
    }
    if (!primitive_tokens.empty()) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::primitive_input);
    }
    if (!context_tokens.empty()) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::context_input);
    }
    if (std::abs(implementation.prediction_error[0U]) > 0.2F) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::prediction_error);
    }

    const FeatureVector empty{};
    const FeatureVector global = implementation.state.recurrent_feedback;
    const FeatureVector message = implementation.state.message_feedback;
    const LayerUpdate ascii = implementation.banks[0U].update(
        *tokenizer_,
        ascii_tokens, empty, global, message, implementation.prediction_error[0U]);
    const LayerUpdate primitive = implementation.banks[1U].update(
        *tokenizer_,
        primitive_tokens, ascii.features, global, message, implementation.prediction_error[0U]);
    const LayerUpdate word = implementation.banks[2U].update(
        *tokenizer_,
        word_tokens, primitive.features, global, message, implementation.prediction_error[0U]);
    const LayerUpdate context = implementation.banks[3U].update(
        *tokenizer_,
        context_tokens, word.features, global, message, implementation.prediction_error[0U]);
    const LayerUpdate grammar = implementation.banks[4U].update(
        *tokenizer_,
        grammar_tokens.empty() ? context_tokens : grammar_tokens,
        context.features, global, message, implementation.prediction_error[0U]);

    implementation.state.layers[0U] = ascii.summary;
    implementation.state.layers[1U] = primitive.summary;
    implementation.state.layers[2U] = word.summary;
    implementation.state.layers[3U] = context.summary;
    implementation.state.layers[4U] = grammar.summary;

    FeatureVector next_recurrent = scaled(grammar.features, config_.feedback_decay);
    add_scaled(next_recurrent, context.features, 0.35F);
    add_scaled(next_recurrent, primitive.features, 0.15F);
    for (std::size_t index = 0U; index < next_recurrent.size(); ++index) {
        implementation.state.recurrent_feedback[index] = clamp_unit(
            0.5F * implementation.state.recurrent_feedback[index] +
            0.5F * std::tanh(next_recurrent[index]));
    }

    FeatureVector next_message = scaled(context.features, 0.5F);
    add_scaled(next_message, grammar.features, 0.5F);
    add_scaled(next_message, implementation.prediction_error, 0.25F);
    for (std::size_t index = 0U; index < next_message.size(); ++index) {
        implementation.state.message_feedback[index] = clamp_unit(
            0.75F * implementation.state.message_feedback[index] +
            0.25F * std::abs(std::tanh(next_message[index])));
    }
    if (std::accumulate(implementation.state.recurrent_feedback.begin(),
                        implementation.state.recurrent_feedback.end(), 0.0F) > 0.01F) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::recurrent_feedback);
    }
    if (std::accumulate(implementation.state.message_feedback.begin(),
                        implementation.state.message_feedback.end(), 0.0F) > 0.01F) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::message_feedback);
    }

    float weighted_novelty = 0.0F;
    float weighted_uncertainty = 0.0F;
    float weighted_center = 0.0F;
    float total_activation = 0.0F;
    for (const LayerPrimitiveSummary& summary : implementation.state.layers) {
        weighted_novelty += summary.novelty;
        weighted_uncertainty += summary.uncertainty;
        weighted_center += summary.center_of_mass;
        total_activation += summary.mean_activation;
    }
    const float layer_scale = 1.0F / static_cast<float>(tokenizer_layer_count);
    auto& values = implementation.state.primitives.values;
    values[0U] = clamp_unit(total_activation * layer_scale);
    values[1U] = clamp_unit(weighted_center * layer_scale);
    values[2U] = clamp_unit(weighted_novelty * layer_scale);
    values[3U] = clamp_unit(weighted_uncertainty * layer_scale);
    values[4U] = clamp_unit(std::abs(implementation.state.recurrent_feedback[0U]));
    values[5U] = clamp_unit(std::abs(implementation.state.message_feedback[0U]));
    values[6U] = implementation.state.layers[3U].upward_signal;
    values[7U] = implementation.state.layers[2U].downward_feedback;
    values[8U] = static_cast<float>(std::count_if(
        implementation.state.layers.begin(), implementation.state.layers.end(),
        [](const LayerPrimitiveSummary& summary) { return summary.mean_activation > 0.05F; })) /
                 static_cast<float>(tokenizer_layer_count);
    values[9U] = clamp_unit(1.0F - values[3U]);
    values[10U] = clamp_unit(std::abs(implementation.prediction_error[0U]));
    values[11U] = clamp_unit(static_cast<float>(implementation.state.step % 1000U) / 1000.0F);
    for (std::size_t region = 0U; region < tokenizer_region_count; ++region) {
        const float region_value = implementation.state.layers[3U].region_activations.empty()
                                       ? 0.0F
                                       : implementation.state.layers[3U].region_activations[
                                             region % implementation.state.layers[3U].region_activations.size()];
        implementation.state.primitives.region_states[region] = static_cast<std::uint8_t>(
            std::clamp(static_cast<int>(std::lround(clamp_unit(region_value) * 3.0F)), 0, 3));
    }
    if (values[2U] > 0.35F) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::novel_input);
    }
    if (values[9U] > 0.65F) {
        implementation.state.primitives.flags |= primitive_bit(PrimitiveFlag::stable_state);
    }
}

void TokenizerNeuralStack::apply_message_feedback(const std::vector<float>& feedback) {
    const std::size_t count = std::min(feedback.size(), tokenizer_feature_count);
    for (std::size_t index = 0U; index < count; ++index) {
        implementation_->state.message_feedback[index] = clamp_unit(
            0.65F * implementation_->state.message_feedback[index] + 0.35F *
            std::abs(feedback[index]));
    }
}

void TokenizerNeuralStack::apply_prediction_error(float normalized_error) {
    const float error = clamp_unit(std::abs(normalized_error));
    implementation_->prediction_error.fill(error);
}

const TokenizerNeuralConfig& TokenizerNeuralStack::config() const noexcept {
    return config_;
}

TokenizerNeuralState TokenizerNeuralStack::state() const {
    return implementation_->state;
}

TokenizerPrimitiveState TokenizerNeuralStack::primitive_state() const {
    return implementation_->state.primitives;
}

bool TokenizerNeuralStack::save(std::ostream& output, std::string& error) const {
    const Impl& implementation = *implementation_;
    for (const LearningNodeBank& bank : implementation.banks) {
        if (!bank.save(output)) {
            error = "could not write tokenizer learning nodes";
            return false;
        }
    }
    if (!write_value(output, implementation.state.step) ||
        !write_value(output, implementation.state.primitives.flags) ||
        !write_features(output, implementation.state.recurrent_feedback) ||
        !write_features(output, implementation.state.message_feedback) ||
        !write_features(output, implementation.prediction_error)) {
        error = "could not write tokenizer feedback state";
        return false;
    }
    output.write(reinterpret_cast<const char*>(implementation.state.primitives.values.data()),
                 static_cast<std::streamsize>(implementation.state.primitives.values.size() * sizeof(float)));
    bits::BitWriter packed_regions;
    for (const std::uint8_t state : implementation.state.primitives.region_states) {
        packed_regions.write(state, 2U);
    }
    const std::uint64_t packed_bit_count = packed_regions.bit_count();
    const std::uint64_t packed_byte_count = packed_regions.bytes().size();
    if (!write_value(output, packed_bit_count) || !write_value(output, packed_byte_count)) {
        error = "could not write packed tokenizer region header";
        return false;
    }
    output.write(reinterpret_cast<const char*>(packed_regions.bytes().data()),
                 static_cast<std::streamsize>(packed_regions.bytes().size()));
    if (!output) {
        error = "could not write tokenizer primitive state";
        return false;
    }
    return true;
}

bool TokenizerNeuralStack::load(std::istream& input, std::string& error) {
    Impl& implementation = *implementation_;
    for (LearningNodeBank& bank : implementation.banks) {
        if (!bank.load(input)) {
            error = "tokenizer checkpoint does not match this learning-node configuration";
            return false;
        }
    }
    if (!read_value(input, implementation.state.step) ||
        !read_value(input, implementation.state.primitives.flags) ||
        !read_features(input, implementation.state.recurrent_feedback) ||
        !read_features(input, implementation.state.message_feedback) ||
        !read_features(input, implementation.prediction_error)) {
        error = "could not read tokenizer feedback state";
        return false;
    }
    input.read(reinterpret_cast<char*>(implementation.state.primitives.values.data()),
               static_cast<std::streamsize>(implementation.state.primitives.values.size() * sizeof(float)));
    std::uint64_t packed_bit_count = 0U;
    std::uint64_t packed_byte_count = 0U;
    constexpr std::size_t expected_bits = tokenizer_region_count * 2U;
    constexpr std::size_t expected_bytes = bits::bytes_for_bits(expected_bits);
    if (!read_value(input, packed_bit_count) || !read_value(input, packed_byte_count) ||
        packed_bit_count != expected_bits || packed_byte_count != expected_bytes) {
        error = "invalid packed tokenizer region state";
        return false;
    }
    std::vector<std::byte> packed_regions(static_cast<std::size_t>(packed_byte_count));
    input.read(reinterpret_cast<char*>(packed_regions.data()),
               static_cast<std::streamsize>(packed_regions.size()));
    if (input) {
        bits::BitReader reader(std::span<const std::byte>(packed_regions), expected_bits);
        for (std::uint8_t& state : implementation.state.primitives.region_states) {
            state = static_cast<std::uint8_t>(reader.read(2U));
        }
    }
    if (!input) {
        error = "could not read tokenizer primitive state";
        return false;
    }
    return true;
}

}  // namespace firstagent::text
