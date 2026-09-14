#include "firstagent/NeuralNetwork.hpp"
#include "firstagent/Parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace firstagent::nn {
namespace {

std::uint64_t mix_seed(std::uint64_t seed, std::uint64_t salt) {
    std::uint64_t value = seed + salt + 0x9E37'79B9'7F4A'7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
    return value ^ (value >> 31U);
}

TransformerConfig validate_config(TransformerConfig config) {
    if (config.vocabulary_size == 0U || config.maximum_sequence_length == 0U ||
        config.model_dimension == 0U || config.layer_count == 0U ||
        config.attention_head_count == 0U || config.key_value_head_count == 0U ||
        config.feed_forward_dimension == 0U) {
        throw std::invalid_argument("Transformer dimensions must be non-zero");
    }
    if (config.model_dimension % config.attention_head_count != 0U) {
        throw std::invalid_argument("model dimension must be divisible by attention head count");
    }
    if (config.attention_head_count % config.key_value_head_count != 0U) {
        throw std::invalid_argument("attention head count must be divisible by key/value head count");
    }
    const std::size_t head_dimension = config.model_dimension / config.attention_head_count;
    if (head_dimension % 2U != 0U) {
        throw std::invalid_argument("attention head dimension must be even for rotary embeddings");
    }
    if (config.rotary_dimension == 0U) {
        config.rotary_dimension = head_dimension;
    }
    if (config.rotary_dimension > head_dimension || config.rotary_dimension % 2U != 0U) {
        throw std::invalid_argument("rotary dimension must be even and no larger than head dimension");
    }
    if (!(config.rope_theta > 1.0F) || !(config.rms_epsilon > 0.0F)) {
        throw std::invalid_argument("Transformer numerical constants are invalid");
    }
    return config;
}

std::size_t sample_logits(const Tensor& logits,
                          const GenerationConfig& generation,
                          std::mt19937_64& generator) {
    if (logits.ndim() != 2U || logits.shape()[0] == 0U || logits.shape()[1] == 0U) {
        throw std::invalid_argument("sampling requires non-empty rank-2 logits");
    }
    const std::size_t vocabulary = logits.shape()[1];
    const std::size_t start = (logits.shape()[0] - 1U) * vocabulary;
    if (generation.temperature <= 0.0F) {
        return argmax_last_row(logits);
    }

    std::vector<std::size_t> candidates(vocabulary, 0U);
    for (std::size_t index = 0U; index < vocabulary; ++index) {
        candidates[index] = index;
    }
    if (generation.top_k > 0U && generation.top_k < vocabulary) {
        const auto comparator = [&logits, start](std::size_t left, std::size_t right) {
            return logits.values()[start + left] > logits.values()[start + right];
        };
        std::partial_sort(candidates.begin(), candidates.begin() +
                          static_cast<std::ptrdiff_t>(generation.top_k), candidates.end(), comparator);
        candidates.resize(generation.top_k);
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (const std::size_t index : candidates) {
        maximum = std::max(maximum, logits.values()[start + index]);
    }
    std::vector<double> probabilities;
    probabilities.reserve(candidates.size());
    for (const std::size_t index : candidates) {
        const float scaled = (logits.values()[start + index] - maximum) / generation.temperature;
        probabilities.push_back(static_cast<double>(std::exp(scaled)));
    }
    std::discrete_distribution<std::size_t> distribution(probabilities.begin(), probabilities.end());
    return candidates[distribution(generator)];
}

std::size_t validated_head_dimension(const TransformerConfig& config) {
    const TransformerConfig validated = validate_config(config);
    return validated.model_dimension / validated.attention_head_count;
}

std::size_t validated_rotary_dimension(const TransformerConfig& config) {
    const TransformerConfig validated = validate_config(config);
    return validated.rotary_dimension;
}

void append_parameters(std::vector<Tensor*>& destination, std::vector<Tensor*> source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

}  // namespace

Linear::Linear(std::size_t input_features,
               std::size_t output_features,
               std::uint64_t seed,
               bool with_bias)
    : input_features_(input_features),
      output_features_(output_features),
      weight_(Tensor::random_normal({input_features, output_features},
                                    0.0F,
                                    1.0F / std::sqrt(static_cast<float>(input_features)),
                                    mix_seed(seed, 0x11ULL),
                                    true)),
      bias_(with_bias ? Tensor::zeros({output_features}, true) : Tensor{}),
      with_bias_(with_bias) {
    if (input_features_ == 0U || output_features_ == 0U) {
        throw std::invalid_argument("Linear dimensions must be non-zero");
    }
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.ndim() != 2U || input.shape()[1] != input_features_) {
        throw std::invalid_argument("Linear input must have shape [rows, input_features]");
    }
    Tensor result = matmul(input, weight_);
    if (with_bias_) {
        result = result + bias_;
    }
    return result;
}

std::vector<Tensor*> Linear::parameters() noexcept {
    if (with_bias_) {
        return {&weight_, &bias_};
    }
    return {&weight_};
}

const Tensor& Linear::weight() const noexcept {
    return weight_;
}

const Tensor& Linear::bias() const noexcept {
    return bias_;
}

Embedding::Embedding(std::size_t vocabulary_size,
                     std::size_t embedding_dimension,
                     std::uint64_t seed)
    : weight_(Tensor::random_normal({vocabulary_size, embedding_dimension},
                                    0.0F,
                                    0.02F,
                                    mix_seed(seed, 0x21ULL),
                                    true)) {
    if (vocabulary_size == 0U || embedding_dimension == 0U) {
        throw std::invalid_argument("Embedding dimensions must be non-zero");
    }
}

Tensor Embedding::forward(const std::vector<std::size_t>& token_ids) const {
    if (token_ids.empty()) {
        throw std::invalid_argument("Embedding cannot process an empty token sequence");
    }
    return embedding(weight_, token_ids);
}

std::vector<Tensor*> Embedding::parameters() noexcept {
    return {&weight_};
}

const Tensor& Embedding::weight() const noexcept {
    return weight_;
}

RMSNorm::RMSNorm(std::size_t features, float epsilon)
    : weight_(Tensor::ones({features}, true)), epsilon_(epsilon) {
    if (features == 0U || !(epsilon > 0.0F)) {
        throw std::invalid_argument("RMSNorm configuration is invalid");
    }
}

Tensor RMSNorm::forward(const Tensor& input) const {
    return rms_norm(input, weight_, epsilon_);
}

std::vector<Tensor*> RMSNorm::parameters() noexcept {
    return {&weight_};
}

SwiGLU::SwiGLU(std::size_t model_dimension,
               std::size_t hidden_dimension,
               std::uint64_t seed)
    : gate_(model_dimension, hidden_dimension, mix_seed(seed, 0x31ULL)),
      value_(model_dimension, hidden_dimension, mix_seed(seed, 0x32ULL)),
      output_(hidden_dimension, model_dimension, mix_seed(seed, 0x33ULL)) {}

Tensor SwiGLU::forward(const Tensor& input) const {
    const Tensor gated = silu(gate_.forward(input));
    const Tensor value = value_.forward(input);
    return output_.forward(gated * value);
}

std::vector<Tensor*> SwiGLU::parameters() noexcept {
    std::vector<Tensor*> result;
    append_parameters(result, gate_.parameters());
    append_parameters(result, value_.parameters());
    append_parameters(result, output_.parameters());
    return result;
}

void KeyValueCache::clear() noexcept {
    key = Tensor{};
    value = Tensor{};
}

std::size_t KeyValueCache::length() const noexcept {
    if (!key.defined() || key.ndim() != 4U) {
        return 0U;
    }
    return key.shape()[2];
}

MultiHeadSelfAttention::MultiHeadSelfAttention(const TransformerConfig& config,
                                               std::uint64_t seed)
    : model_dimension_(config.model_dimension),
      query_head_count_(config.attention_head_count),
      key_value_head_count_(config.key_value_head_count),
      head_dimension_(validated_head_dimension(config)),
      rotary_dimension_(validated_rotary_dimension(config)),
      rope_theta_(config.rope_theta),
      query_(config.model_dimension,
             config.attention_head_count * validated_head_dimension(config),
             mix_seed(seed, 0x41ULL)),
      key_(config.model_dimension,
           config.key_value_head_count * validated_head_dimension(config),
           mix_seed(seed, 0x42ULL)),
      value_(config.model_dimension,
             config.key_value_head_count * validated_head_dimension(config),
             mix_seed(seed, 0x43ULL)),
      output_(config.attention_head_count * validated_head_dimension(config),
              config.model_dimension,
              mix_seed(seed, 0x44ULL)) {
    (void)validate_config(config);
}

Tensor MultiHeadSelfAttention::forward(const Tensor& input) const {
    if (input.ndim() != 2U || input.shape()[1] != model_dimension_ || input.shape()[0] == 0U) {
        throw std::invalid_argument("self-attention input must have shape [sequence, model_dimension]");
    }
    Tensor query = split_attention_heads(query_.forward(input), query_head_count_);
    Tensor key = split_attention_heads(key_.forward(input), key_value_head_count_);
    Tensor value = split_attention_heads(value_.forward(input), key_value_head_count_);
    query = apply_rope(query, rotary_dimension_, rope_theta_);
    key = apply_rope(key, rotary_dimension_, rope_theta_);
    const Tensor attended = scaled_dot_product_attention(query, key, value, true);
    return output_.forward(merge_attention_heads(attended));
}

Tensor MultiHeadSelfAttention::forward_step(const Tensor& input,
                                            KeyValueCache& cache,
                                            std::size_t position) const {
    if (input.ndim() != 2U || input.shape()[0] != 1U || input.shape()[1] != model_dimension_) {
        throw std::invalid_argument("attention step input must have shape [1, model_dimension]");
    }
    Tensor query = split_attention_heads(query_.forward(input), query_head_count_);
    Tensor key = split_attention_heads(key_.forward(input), key_value_head_count_);
    Tensor value = split_attention_heads(value_.forward(input), key_value_head_count_);
    query = apply_rope(query, rotary_dimension_, rope_theta_, position);
    key = apply_rope(key, rotary_dimension_, rope_theta_, position);

    cache.key = concatenate_sequence(cache.key, key);
    cache.value = concatenate_sequence(cache.value, value);
    const Tensor attended = scaled_dot_product_attention(query, cache.key, cache.value, true, position);
    return output_.forward(merge_attention_heads(attended));
}

std::vector<Tensor*> MultiHeadSelfAttention::parameters() noexcept {
    std::vector<Tensor*> result;
    append_parameters(result, query_.parameters());
    append_parameters(result, key_.parameters());
    append_parameters(result, value_.parameters());
    append_parameters(result, output_.parameters());
    return result;
}

TransformerBlock::TransformerBlock(const TransformerConfig& config, std::uint64_t seed)
    : attention_norm_(config.model_dimension, config.rms_epsilon),
      attention_(config, mix_seed(seed, 0x51ULL)),
      feed_forward_norm_(config.model_dimension, config.rms_epsilon),
      feed_forward_(config.model_dimension,
                    config.feed_forward_dimension,
                    mix_seed(seed, 0x52ULL)) {}

Tensor TransformerBlock::forward(const Tensor& input) const {
    const Tensor attention_output = attention_.forward(attention_norm_.forward(input));
    const Tensor attention_residual = input + attention_output;
    const Tensor feed_forward_output = feed_forward_.forward(feed_forward_norm_.forward(attention_residual));
    return attention_residual + feed_forward_output;
}

Tensor TransformerBlock::forward_step(const Tensor& input,
                                      KeyValueCache& cache,
                                      std::size_t position) const {
    const Tensor attention_output =
        attention_.forward_step(attention_norm_.forward(input), cache, position);
    const Tensor attention_residual = input + attention_output;
    const Tensor feed_forward_output = feed_forward_.forward(feed_forward_norm_.forward(attention_residual));
    return attention_residual + feed_forward_output;
}

std::vector<Tensor*> TransformerBlock::parameters() noexcept {
    std::vector<Tensor*> result;
    append_parameters(result, attention_norm_.parameters());
    append_parameters(result, attention_.parameters());
    append_parameters(result, feed_forward_norm_.parameters());
    append_parameters(result, feed_forward_.parameters());
    return result;
}

TransformerModel::TransformerModel(TransformerConfig config)
    : config_(validate_config(std::move(config))),
      token_embedding_(config_.vocabulary_size, config_.model_dimension, mix_seed(config_.seed, 0x61ULL)),
      blocks_(),
      final_norm_(config_.model_dimension, config_.rms_epsilon) {
    blocks_.reserve(config_.layer_count);
    for (std::size_t layer = 0U; layer < config_.layer_count; ++layer) {
        blocks_.emplace_back(config_, mix_seed(config_.seed, 0x100ULL + layer));
    }
}

Tensor TransformerModel::project_to_vocabulary(const Tensor& hidden) const {
    // Weight tying shares the token embedding matrix with the output head.
    return matmul(hidden, token_embedding_.weight().transpose());
}

Tensor TransformerModel::forward(const std::vector<std::size_t>& token_ids) const {
    if (token_ids.empty() || token_ids.size() > config_.maximum_sequence_length) {
        throw std::invalid_argument("input sequence is empty or exceeds the model context length");
    }
    Tensor hidden = token_embedding_.forward(token_ids);
    for (const TransformerBlock& block : blocks_) {
        hidden = block.forward(hidden);
    }
    return project_to_vocabulary(final_norm_.forward(hidden));
}

Tensor TransformerModel::loss(const std::vector<std::size_t>& input_ids,
                              const std::vector<std::size_t>& target_ids) const {
    if (input_ids.size() != target_ids.size() || input_ids.empty()) {
        throw std::invalid_argument("input and target sequences must be non-empty and equally sized");
    }
    return cross_entropy(forward(input_ids), target_ids);
}

Tensor TransformerModel::step(std::size_t token_id,
                              std::vector<KeyValueCache>& cache,
                              std::size_t position) const {
    if (token_id >= config_.vocabulary_size || position >= config_.maximum_sequence_length ||
        cache.size() != blocks_.size()) {
        throw std::invalid_argument("invalid Transformer decoding step");
    }
    Tensor hidden = token_embedding_.forward({token_id});
    for (std::size_t layer = 0U; layer < blocks_.size(); ++layer) {
        hidden = blocks_[layer].forward_step(hidden, cache[layer], position);
    }
    return project_to_vocabulary(final_norm_.forward(hidden));
}

std::vector<std::size_t> TransformerModel::generate(const std::vector<std::size_t>& prompt,
                                                    const GenerationConfig& generation) const {
    if (prompt.empty() || prompt.size() > config_.maximum_sequence_length ||
        generation.maximum_new_tokens > config_.maximum_sequence_length - prompt.size()) {
        throw std::invalid_argument("generation request exceeds the model context length");
    }
    if (generation.temperature < 0.0F) {
        throw std::invalid_argument("generation temperature cannot be negative");
    }

    NoGradGuard no_grad;
    std::mt19937_64 generator(generation.seed);
    std::vector<KeyValueCache> cache(blocks_.size());
    std::vector<std::size_t> result = prompt;
    Tensor logits;
    std::size_t position = 0U;
    for (const std::size_t token : prompt) {
        logits = step(token, cache, position);
        ++position;
    }
    for (std::size_t generated = 0U; generated < generation.maximum_new_tokens; ++generated) {
        const std::size_t token = sample_logits(logits, generation, generator);
        result.push_back(token);
        logits = step(token, cache, position);
        ++position;
    }
    return result;
}

std::vector<Tensor*> TransformerModel::parameters() noexcept {
    std::vector<Tensor*> result;
    append_parameters(result, token_embedding_.parameters());
    for (TransformerBlock& block : blocks_) {
        append_parameters(result, block.parameters());
    }
    append_parameters(result, final_norm_.parameters());
    return result;
}

std::size_t TransformerModel::parameter_count() const noexcept {
    std::size_t count = token_embedding_.weight().numel();
    const std::size_t head_dimension = config_.model_dimension / config_.attention_head_count;
    const std::size_t query_width = config_.attention_head_count * head_dimension;
    const std::size_t key_value_width = config_.key_value_head_count * head_dimension;
    const std::size_t attention_parameters =
        config_.model_dimension * query_width + query_width +
        (config_.model_dimension * key_value_width + key_value_width) * 2U +
        query_width * config_.model_dimension + config_.model_dimension;
    const std::size_t feed_forward_parameters =
        (config_.model_dimension * config_.feed_forward_dimension + config_.feed_forward_dimension) * 2U +
        config_.feed_forward_dimension * config_.model_dimension + config_.model_dimension;
    const std::size_t normalization_parameters = config_.model_dimension * 2U;
    count += config_.layer_count *
             (attention_parameters + feed_forward_parameters + normalization_parameters);
    count += config_.model_dimension;
    return count;
}

const TransformerConfig& TransformerModel::config() const noexcept {
    return config_;
}

AdamW::AdamW(std::vector<Tensor*> parameters,
             float learning_rate,
             float beta1,
             float beta2,
             float epsilon,
             float weight_decay)
    : parameters_(std::move(parameters)),
      first_moment_(),
      second_moment_(),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay) {
    if (!(learning_rate_ > 0.0F) || !(beta1_ >= 0.0F && beta1_ < 1.0F) ||
        !(beta2_ >= 0.0F && beta2_ < 1.0F) || !(epsilon_ > 0.0F) || weight_decay_ < 0.0F) {
        throw std::invalid_argument("AdamW hyperparameters are invalid");
    }
    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());
    for (const Tensor* parameter : parameters_) {
        if (parameter == nullptr || !parameter->requires_grad()) {
            throw std::invalid_argument("AdamW requires non-null trainable parameters");
        }
        first_moment_.emplace_back(parameter->numel(), 0.0F);
        second_moment_.emplace_back(parameter->numel(), 0.0F);
    }
}

void AdamW::step() {
    ++update_count_;
    const float first_correction = 1.0F - std::pow(beta1_, static_cast<float>(update_count_));
    const float second_correction = 1.0F - std::pow(beta2_, static_cast<float>(update_count_));
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 1U;
    parallel::parallel_for(0U, parameters_.size(), [&](std::size_t parameter_index) {
        Tensor& parameter = *parameters_[parameter_index];
        const auto& gradient = parameter.grad_values();
        auto& first = first_moment_[parameter_index];
        auto& second = second_moment_[parameter_index];
        for (std::size_t value_index = 0U; value_index < parameter.numel(); ++value_index) {
            first[value_index] = beta1_ * first[value_index] + (1.0F - beta1_) * gradient[value_index];
            second[value_index] = beta2_ * second[value_index] +
                                  (1.0F - beta2_) * gradient[value_index] * gradient[value_index];
            const float first_hat = first[value_index] / first_correction;
            const float second_hat = second[value_index] / second_correction;
            const float adaptive_update = first_hat / (std::sqrt(second_hat) + epsilon_);
            parameter.mutable_values()[value_index] *= 1.0F - learning_rate_ * weight_decay_;
            parameter.mutable_values()[value_index] -= learning_rate_ * adaptive_update;
        }
    }, parallel_config);
}

void AdamW::zero_grad() {
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 1U;
    parallel::parallel_for(0U, parameters_.size(), [&](std::size_t parameter_index) {
        parameters_[parameter_index]->zero_grad();
    }, parallel_config);
}

std::size_t AdamW::update_count() const noexcept {
    return update_count_;
}

AdamW::State AdamW::state() const {
    return State{
        .update_count = update_count_,
        .first_moment = first_moment_,
        .second_moment = second_moment_,
    };
}

void AdamW::load_state(State state) {
    if (state.first_moment.size() != parameters_.size() ||
        state.second_moment.size() != parameters_.size()) {
        throw std::invalid_argument("AdamW state parameter count does not match optimizer");
    }
    for (std::size_t index = 0U; index < parameters_.size(); ++index) {
        if (state.first_moment[index].size() != parameters_[index]->numel() ||
            state.second_moment[index].size() != parameters_[index]->numel()) {
            throw std::invalid_argument("AdamW state tensor size does not match optimizer");
        }
    }
    update_count_ = state.update_count;
    first_moment_ = std::move(state.first_moment);
    second_moment_ = std::move(state.second_moment);
}

}  // namespace firstagent::nn
