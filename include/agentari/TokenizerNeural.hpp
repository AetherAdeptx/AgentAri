#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <iosfwd>
#include <string_view>
#include <vector>

namespace agentari::text {

using TokenId = std::uint32_t;

class LayeredTokenizer;

inline constexpr std::size_t tokenizer_layer_count = 5U;
inline constexpr std::size_t tokenizer_feature_count = 16U;
inline constexpr std::size_t tokenizer_primitive_value_count = 12U;
inline constexpr std::size_t tokenizer_region_count = 16U;

struct TokenizerNeuralConfig {
    // These are bounded learning-node banks, not one heavyweight parameter
    // vector per vocabulary entry. Hashing lets every tokenizer capacity share
    // a manageable adaptive representation.
    std::size_t ascii_node_count{256U};
    std::size_t primitive_node_count{1024U};
    std::size_t word_node_count{4096U};
    std::size_t context_node_count{2048U};
    std::size_t grammar_node_count{256U};
    std::size_t expert_count{4U};
    std::size_t region_count{tokenizer_region_count};
    float ascii_learning_factor{0.04F};
    float primitive_learning_factor{0.035F};
    float word_learning_factor{0.025F};
    float context_learning_factor{0.02F};
    float grammar_learning_factor{0.015F};
    float activation_decay{0.92F};
    float trace_decay{0.96F};
    float feedback_decay{0.85F};
};

enum class PrimitiveFlag : std::uint8_t {
    ascii_input = 0U,
    primitive_input = 1U,
    unknown_word = 2U,
    symbol_input = 3U,
    context_input = 4U,
    prediction_error = 5U,
    message_feedback = 6U,
    recurrent_feedback = 7U,
    novel_input = 8U,
    stable_state = 9U,
};

struct LayerPrimitiveSummary {
    std::size_t node_count{0U};
    float total_activation{0.0F};
    float mean_activation{0.0F};
    float center_of_mass{0.0F};
    float entropy{0.0F};
    float novelty{0.0F};
    float uncertainty{1.0F};
    float upward_signal{0.0F};
    float downward_feedback{0.0F};
    std::vector<float> region_activations;
    std::vector<float> expert_activations;
    std::size_t active_expert{0U};
    std::array<float, tokenizer_feature_count> features{};
};

struct TokenizerPrimitiveState {
    std::uint64_t flags{0U};
    std::array<float, tokenizer_primitive_value_count> values{};
    std::array<std::uint8_t, tokenizer_region_count> region_states{};
};

struct TokenizerNeuralState {
    std::uint64_t step{0U};
    std::array<LayerPrimitiveSummary, tokenizer_layer_count> layers;
    TokenizerPrimitiveState primitives;
    std::array<float, tokenizer_feature_count> recurrent_feedback{};
    std::array<float, tokenizer_feature_count> message_feedback{};
};

class TokenizerNeuralStack {
public:
    TokenizerNeuralStack(const LayeredTokenizer& tokenizer,
                         TokenizerNeuralConfig config = {});
    ~TokenizerNeuralStack();

    TokenizerNeuralStack(const TokenizerNeuralStack&) = delete;
    TokenizerNeuralStack& operator=(const TokenizerNeuralStack&) = delete;

    void observe_text(std::string_view input);
    void observe_tokens(const std::vector<TokenId>& tokens);
    void observe(const std::vector<TokenId>& word_tokens,
                 const std::vector<TokenId>& primitive_tokens,
                 const std::vector<TokenId>& ascii_tokens,
                 const std::vector<TokenId>& context_tokens,
                 const std::vector<TokenId>& grammar_tokens = {});

    // External message feedback is deliberately separate from the internal
    // recurrent loop so an agent response can modulate the next input cycle.
    void apply_message_feedback(const std::vector<float>& feedback);
    void apply_prediction_error(float normalized_error);
    [[nodiscard]] bool save(std::ostream& output, std::string& error) const;
    [[nodiscard]] bool load(std::istream& input, std::string& error);

    [[nodiscard]] const TokenizerNeuralConfig& config() const noexcept;
    [[nodiscard]] TokenizerNeuralState state() const;
    [[nodiscard]] TokenizerPrimitiveState primitive_state() const;

private:
    struct Impl;
    const LayeredTokenizer* tokenizer_;
    TokenizerNeuralConfig config_;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace agentari::text
