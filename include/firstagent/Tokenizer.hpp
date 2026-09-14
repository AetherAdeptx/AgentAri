#pragma once

#include "firstagent/NeuralNetwork.hpp"
#include "firstagent/TokenizerNeural.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace firstagent::text {

enum class TokenLayer : std::uint8_t {
    ascii = 1U,
    primitive = 2U,
    word = 3U,
    context = 4U,
    grammar = 5U,
};

struct TokenRecord {
    TokenId id{0U};
    TokenLayer layer{TokenLayer::ascii};
    std::string text;
    std::string semantic_role;
    std::size_t rank{0U};
    std::uint8_t depth{128U};
    float mobility{0.5F};
    bool prefix_only{false};
    bool suffix_only{false};
};

struct TokenizerConfig {
    // Layer 1: 256 active ASCII byte tokens plus 256 reserved growth slots.
    std::size_t ascii_base_capacity{256U};
    std::size_t ascii_expansion_capacity{256U};
    // Layer 2: 1,024 active word-building rules plus 1,024 expansion slots.
    std::size_t primitive_rule_capacity{1024U};
    std::size_t primitive_expansion_capacity{1024U};
    // Layer 3: initial word index size, with room to grow explicitly later.
    std::size_t vocabulary_capacity{100000U};
};

class LayeredTokenizer {
public:
    explicit LayeredTokenizer(TokenizerConfig config = {});

    [[nodiscard]] const TokenizerConfig& config() const noexcept;
    [[nodiscard]] std::size_t primitive_count() const noexcept;
    [[nodiscard]] std::size_t primitive_capacity() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t vocabulary_capacity() const noexcept;
    [[nodiscard]] std::size_t remaining_vocabulary_slots() const noexcept;
    [[nodiscard]] std::size_t word_token_count() const noexcept;
    [[nodiscard]] std::size_t token_count() const noexcept;
    [[nodiscard]] std::size_t model_vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t word_index(TokenId token) const noexcept;
    [[nodiscard]] TokenId token_from_word_index(std::size_t index) const noexcept;
    [[nodiscard]] bool is_known_word_token(TokenId token) const noexcept;
    [[nodiscard]] float token_mobility(TokenId token) const noexcept;

    [[nodiscard]] TokenId add_primitive(std::string primitive,
                                        std::string semantic_role = {},
                                        std::uint8_t depth = 160U);
    [[nodiscard]] TokenId add_word(std::string word, std::size_t rank = 0U);
    [[nodiscard]] bool load_primitive_file(const std::string& path,
                                           std::size_t maximum_primitives,
                                           std::string& error);
    [[nodiscard]] bool load_vocabulary_file(const std::string& path,
                                             std::size_t maximum_words,
                                             std::string& error);
    [[nodiscard]] bool load_depth_map_file(const std::string& path, std::string& error);

    [[nodiscard]] TokenId lookup_word(std::string_view word) const noexcept;
    [[nodiscard]] const TokenRecord& record(TokenId id) const;
    [[nodiscard]] std::string_view text(TokenId id) const;

    // Word-level encoding is used by layer 4. Unknown words use <unk>.
    [[nodiscard]] std::vector<TokenId> encode_words(std::string_view input) const;
    [[nodiscard]] std::string decode_words(const std::vector<TokenId>& tokens) const;

    // Known words remain one layer-3 token. Unknown words are decomposed into
    // the longest registered primitives and ASCII bytes for residual spelling.
    [[nodiscard]] std::vector<TokenId> encode_word_parts(std::string_view word) const;
    [[nodiscard]] std::string decode_word_parts(const std::vector<TokenId>& tokens) const;

    // Layer 1 fallback: each byte maps directly to [0, 255].
    [[nodiscard]] static std::vector<TokenId> encode_ascii(std::string_view input);
    // Mixed encoding retains separators and symbols as byte tokens while
    // keeping known words as shallow word-layer tokens.
    [[nodiscard]] std::vector<TokenId> encode_text(std::string_view input) const;
    [[nodiscard]] std::string decode_text(const std::vector<TokenId>& tokens) const;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;

private:
    TokenizerConfig config_;
    std::vector<TokenRecord> records_;
    std::unordered_map<std::string, TokenId> primitive_ids_;
    std::unordered_map<std::string, TokenId> word_ids_;
    std::vector<TokenId> word_tokens_;
    std::unordered_map<TokenId, std::size_t> word_indices_;
    TokenId unknown_word_id_{0U};

    static std::string normalize_word(std::string_view word);
    static float mobility_for_depth(std::uint8_t depth) noexcept;
};

struct Prediction {
    TokenId token{0U};
    std::string word;
    std::uint64_t count{0U};
    float probability{0.0F};
};

struct GrammarConfig {
    bool normalize_spacing{true};
    bool capitalize_sentences{true};
    bool append_terminal_punctuation{true};
    std::size_t maximum_text_bytes{1U << 20U};
};

struct GrammarState {
    std::uint64_t observations{0U};
    std::uint64_t observed_tokens{0U};
    std::uint64_t observed_sentences{0U};
    std::uint64_t learned_transitions{0U};
};

class GrammarEngine {
public:
    GrammarEngine(const LayeredTokenizer& tokenizer, GrammarConfig config = {});

    void observe(std::string_view text);
    void add_exception(std::string source, std::string replacement);

    [[nodiscard]] std::string repair(std::string_view text) const;
    [[nodiscard]] GrammarState state() const noexcept;

private:
    const LayeredTokenizer* tokenizer_;
    GrammarConfig config_;
    GrammarState state_;
    std::unordered_map<std::string, std::string> exceptions_;
    std::unordered_map<std::string, std::uint64_t> transitions_;
};

struct PredictorConfig {
    std::size_t maximum_context_words{1024U};
    // The tokenizer may hold 100k words, while the first CPU predictor trains
    // against a configurable frequent-word shortlist. Increase this when a
    // sampled/adaptive output head is available.
    std::size_t model_vocabulary_limit{8192U};
    std::size_t model_dimension{64U};
    std::size_t layer_count{2U};
    std::size_t attention_head_count{4U};
    std::size_t key_value_head_count{2U};
    std::size_t feed_forward_dimension{256U};
    std::size_t maximum_sequence_length{1024U};
    std::size_t training_repeats_per_observation{1U};
    std::size_t gradient_accumulation_steps{1U};
    std::size_t maximum_training_windows_per_observation{8U};
    std::size_t training_window_stride{0U};
    float learning_rate{3.0e-4F};
    float weight_decay{0.01F};
    std::uint64_t seed{0xA17E'5EED'2026'0913ULL};
    TokenizerNeuralConfig tokenizer_neural{};
    GrammarConfig grammar{};
    bool grammar_postprocess{true};
};

struct ContextSteeringConfig {
    std::size_t local_window_words{256U};
    std::size_t history_sample_count{256U};
    std::size_t neighborhood_radius{4U};
    std::size_t frame_word_count{4096U};
    std::size_t salient_words_per_frame{8U};
    std::size_t retained_frame_count{32U};
};

struct ContextFrame {
    std::uint64_t log_number{0U};
    std::size_t frame_index{0U};
    std::vector<TokenId> salient_words;
};

struct ContextWindow {
    std::uint64_t log_number{0U};
    std::vector<TokenId> goal_words;
    std::vector<TokenId> frame_words;
    std::vector<TokenId> sampled_history_words;
    std::vector<TokenId> recent_words;

    [[nodiscard]] std::vector<TokenId> flatten(std::size_t maximum_words = 0U) const;
};

class ContextSteering {
public:
    ContextSteering(const LayeredTokenizer& tokenizer, ContextSteeringConfig config = {});

    void record_chat(std::uint64_t log_number, const std::vector<TokenId>& words);
    void record_chat_text(std::uint64_t log_number, std::string_view text);
    void set_goal(std::uint64_t log_number, const std::vector<TokenId>& words);
    void set_goal_text(std::uint64_t log_number, std::string_view text);

    [[nodiscard]] ContextWindow build(std::uint64_t log_number,
                                      const std::vector<TokenId>& current_words) const;
    [[nodiscard]] ContextWindow build_text(std::uint64_t log_number,
                                            std::string_view current_text) const;
    [[nodiscard]] std::size_t frame_count() const noexcept;

private:
    const LayeredTokenizer* tokenizer_;
    ContextSteeringConfig config_;
    std::map<std::uint64_t, std::vector<TokenId>> logs_;
    std::map<std::uint64_t, std::vector<TokenId>> goals_;
    std::deque<ContextFrame> frames_;

    void rebuild_frames();
    [[nodiscard]] std::vector<TokenId> flatten_history(std::uint64_t log_number) const;
    [[nodiscard]] std::vector<TokenId> sample_history(const std::vector<TokenId>& history,
                                                      std::uint64_t log_number) const;
};

class WordPredictor {
public:
    WordPredictor(const LayeredTokenizer& tokenizer, PredictorConfig config = {});

    void observe(const std::vector<TokenId>& tokens);
    void observe_text(std::string_view input);
    [[nodiscard]] bool observe_file(const std::string& path,
                                    std::size_t maximum_bytes,
                                    std::string& error);

    [[nodiscard]] std::vector<Prediction> predict(const std::vector<TokenId>& context,
                                                   std::size_t top_k = 5U) const;
    [[nodiscard]] std::vector<Prediction> predict(const ContextWindow& context,
                                                   std::size_t top_k = 5U) const;
    [[nodiscard]] std::vector<Prediction> predict_text(std::string_view context,
                                                       std::size_t top_k = 5U) const;
    [[nodiscard]] std::string generate(std::string_view prompt,
                                       std::size_t maximum_new_words = 16U,
                                       std::size_t top_k = 1U) const;
    [[nodiscard]] float evaluate_loss(std::string_view text) const;
    [[nodiscard]] std::size_t training_steps() const noexcept;
    [[nodiscard]] float last_loss() const noexcept;
    [[nodiscard]] std::size_t active_vocabulary_size() const noexcept;
    [[nodiscard]] TokenizerNeuralState tokenizer_state() const;
    [[nodiscard]] GrammarState grammar_state() const noexcept;
    void apply_message_feedback(const std::vector<float>& feedback);
    void apply_prediction_error(float normalized_error);
    [[nodiscard]] bool save_checkpoint(const std::string& path, std::string& error);
    [[nodiscard]] bool load_checkpoint(const std::string& path, std::string& error);
    [[nodiscard]] std::size_t maximum_context_words() const noexcept;

private:
    const LayeredTokenizer* tokenizer_;
    PredictorConfig config_;
    mutable GrammarEngine grammar_;
    mutable TokenizerNeuralStack tokenizer_neural_;
    nn::TransformerModel model_;
    nn::AdamW optimizer_;
    std::size_t training_steps_{0U};
    float last_loss_{0.0F};

    void train_model_tokens(const std::vector<TokenId>& tokens);
    [[nodiscard]] std::vector<std::size_t> to_model_ids(const std::vector<TokenId>& tokens) const;
    [[nodiscard]] std::vector<TokenId> trim_context(const std::vector<TokenId>& tokens) const;
};

}  // namespace firstagent::text
