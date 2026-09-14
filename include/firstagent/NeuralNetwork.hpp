#pragma once

#include "firstagent/Tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace firstagent::nn {

class Linear {
public:
    Linear(std::size_t input_features,
           std::size_t output_features,
           std::uint64_t seed,
           bool with_bias = true);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;
    [[nodiscard]] const Tensor& weight() const noexcept;
    [[nodiscard]] const Tensor& bias() const noexcept;

private:
    std::size_t input_features_;
    std::size_t output_features_;
    Tensor weight_;
    Tensor bias_;
    bool with_bias_;
};

class Embedding {
public:
    Embedding(std::size_t vocabulary_size, std::size_t embedding_dimension, std::uint64_t seed);

    [[nodiscard]] Tensor forward(const std::vector<std::size_t>& token_ids) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;
    [[nodiscard]] const Tensor& weight() const noexcept;

private:
    Tensor weight_;
};

class RMSNorm {
public:
    RMSNorm(std::size_t features, float epsilon);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;

private:
    Tensor weight_;
    float epsilon_;
};

class SwiGLU {
public:
    SwiGLU(std::size_t model_dimension,
           std::size_t hidden_dimension,
           std::uint64_t seed);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;

private:
    Linear gate_;
    Linear value_;
    Linear output_;
};

struct TransformerConfig {
    std::size_t vocabulary_size{256U};
    std::size_t maximum_sequence_length{256U};
    std::size_t model_dimension{128U};
    std::size_t layer_count{2U};
    std::size_t attention_head_count{4U};
    std::size_t key_value_head_count{2U};
    std::size_t feed_forward_dimension{512U};
    std::size_t rotary_dimension{0U};
    float rope_theta{10000.0F};
    float rms_epsilon{1.0e-5F};
    std::uint64_t seed{0xF1A5'7EED'1234'5678ULL};
};

struct KeyValueCache {
    Tensor key;
    Tensor value;

    void clear() noexcept;
    [[nodiscard]] std::size_t length() const noexcept;
};

class MultiHeadSelfAttention {
public:
    MultiHeadSelfAttention(const TransformerConfig& config, std::uint64_t seed);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] Tensor forward_step(const Tensor& input,
                                      KeyValueCache& cache,
                                      std::size_t position) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;

private:
    std::size_t model_dimension_;
    std::size_t query_head_count_;
    std::size_t key_value_head_count_;
    std::size_t head_dimension_;
    std::size_t rotary_dimension_;
    float rope_theta_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;
};

class TransformerBlock {
public:
    TransformerBlock(const TransformerConfig& config, std::uint64_t seed);

    [[nodiscard]] Tensor forward(const Tensor& input) const;
    [[nodiscard]] Tensor forward_step(const Tensor& input,
                                      KeyValueCache& cache,
                                      std::size_t position) const;
    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;

private:
    RMSNorm attention_norm_;
    MultiHeadSelfAttention attention_;
    RMSNorm feed_forward_norm_;
    SwiGLU feed_forward_;
};

struct GenerationConfig {
    std::size_t maximum_new_tokens{32U};
    float temperature{0.8F};
    std::size_t top_k{40U};
    std::uint64_t seed{0xC0FFEEULL};
};

class TransformerModel {
public:
    explicit TransformerModel(TransformerConfig config);

    [[nodiscard]] Tensor forward(const std::vector<std::size_t>& token_ids) const;
    [[nodiscard]] Tensor loss(const std::vector<std::size_t>& input_ids,
                              const std::vector<std::size_t>& target_ids) const;
    [[nodiscard]] Tensor step(std::size_t token_id,
                              std::vector<KeyValueCache>& cache,
                              std::size_t position) const;
    [[nodiscard]] std::vector<std::size_t> generate(const std::vector<std::size_t>& prompt,
                                                    const GenerationConfig& generation) const;

    [[nodiscard]] std::vector<Tensor*> parameters() noexcept;
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    [[nodiscard]] const TransformerConfig& config() const noexcept;

private:
    TransformerConfig config_;
    Embedding token_embedding_;
    std::vector<TransformerBlock> blocks_;
    RMSNorm final_norm_;

    [[nodiscard]] Tensor project_to_vocabulary(const Tensor& hidden) const;
};

class AdamW {
public:
    struct State {
        std::size_t update_count{0U};
        std::vector<std::vector<float>> first_moment;
        std::vector<std::vector<float>> second_moment;
    };
    AdamW(std::vector<Tensor*> parameters,
          float learning_rate = 3.0e-4F,
          float beta1 = 0.9F,
          float beta2 = 0.999F,
          float epsilon = 1.0e-8F,
          float weight_decay = 0.01F);

    void step();
    void zero_grad();
    [[nodiscard]] std::size_t update_count() const noexcept;
    [[nodiscard]] State state() const;
    void load_state(State state);

private:
    std::vector<Tensor*> parameters_;
    std::vector<std::vector<float>> first_moment_;
    std::vector<std::vector<float>> second_moment_;
    float learning_rate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    std::size_t update_count_{0U};
};

}  // namespace firstagent::nn
