#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace agentari::neuron {

// Storage policies let a network choose between fast FP32 weights, compact
// int8 weights, and bit-packed binary weights without changing its topology.
// DenseStorage is the default for learning; the compact policies are useful
// for inference or low-memory specialist banks.
template <std::size_t Count>
struct alignas(64U) DenseStorage {
    std::array<float, Count> values{};

    static constexpr std::size_t element_count = Count;
    static constexpr std::size_t storage_bytes = sizeof(DenseStorage);

    [[nodiscard]] float get(std::size_t index) const noexcept { return values[index]; }
    void set(std::size_t index, float value) noexcept { values[index] = value; }
    void add(std::size_t index, float value) noexcept { values[index] += value; }
};

template <std::size_t Count>
struct alignas(16U) QuantizedInt8Storage {
    std::array<std::int8_t, Count> values{};
    float scale{1.0F / 127.0F};

    static constexpr std::size_t element_count = Count;
    static constexpr std::size_t storage_bytes = sizeof(QuantizedInt8Storage);

    [[nodiscard]] float get(std::size_t index) const noexcept {
        return static_cast<float>(values[index]) * scale;
    }

    void set(std::size_t index, float value) noexcept {
        const float normalized = scale > 0.0F ? value / scale : 0.0F;
        const float clamped = std::clamp(normalized, -127.0F, 127.0F);
        values[index] = static_cast<std::int8_t>(std::lround(clamped));
    }

    void add(std::size_t index, float value) noexcept {
        set(index, get(index) + value);
    }

    void set_scale(float new_scale) {
        if (!(new_scale > 0.0F) || !std::isfinite(new_scale)) {
            throw std::invalid_argument("quantized neuron scale must be finite and positive");
        }
        std::array<float, Count> decoded{};
        for (std::size_t index = 0U; index < Count; ++index) {
            decoded[index] = get(index);
        }
        scale = new_scale;
        for (std::size_t index = 0U; index < Count; ++index) {
            set(index, decoded[index]);
        }
    }
};

template <std::size_t Count>
struct alignas(8U) BinaryStorage {
    static constexpr std::size_t word_count = (Count + 63U) / 64U;
    std::array<std::uint64_t, word_count> bits{};

    static constexpr std::size_t element_count = Count;
    static constexpr std::size_t storage_bytes = sizeof(BinaryStorage);

    // Binary weights are represented as -1/+1 so they remain useful in a
    // dot product while consuming one bit each.
    [[nodiscard]] float get(std::size_t index) const noexcept {
        const bool enabled = (bits[index / 64U] & (std::uint64_t{1U} << (index % 64U))) != 0U;
        return enabled ? 1.0F : -1.0F;
    }

    void set(std::size_t index, float value) noexcept {
        const std::uint64_t mask = std::uint64_t{1U} << (index % 64U);
        if (value >= 0.0F) {
            bits[index / 64U] |= mask;
        } else {
            bits[index / 64U] &= ~mask;
        }
    }

    void add(std::size_t index, float value) noexcept {
        set(index, get(index) + value);
    }
};

struct LinearActivation {
    static float apply(float value) noexcept { return value; }
    static float derivative(float) noexcept { return 1.0F; }
};

struct ReluActivation {
    static float apply(float value) noexcept { return std::max(0.0F, value); }
    static float derivative(float value) noexcept { return value > 0.0F ? 1.0F : 0.0F; }
};

struct TanhActivation {
    static float apply(float value) noexcept { return std::tanh(value); }
    static float derivative(float value) noexcept {
        const float output = std::tanh(value);
        return 1.0F - output * output;
    }
};

struct SigmoidActivation {
    static float apply(float value) noexcept {
        return 1.0F / (1.0F + std::exp(-value));
    }
    static float derivative(float value) noexcept {
        const float output = apply(value);
        return output * (1.0F - output);
    }
};

struct SiluActivation {
    static float apply(float value) noexcept {
        return value * SigmoidActivation::apply(value);
    }
    static float derivative(float value) noexcept {
        const float sigmoid = SigmoidActivation::apply(value);
        return sigmoid * (1.0F + value * (1.0F - sigmoid));
    }
};

struct GeluActivation {
    static float apply(float value) noexcept {
        constexpr float coefficient = 0.044715F;
        constexpr float root_two_over_pi = 0.7978845608F;
        const float inner = root_two_over_pi *
                            (value + coefficient * value * value * value);
        return 0.5F * value * (1.0F + std::tanh(inner));
    }
    static float derivative(float value) noexcept {
        constexpr float coefficient = 0.044715F;
        constexpr float root_two_over_pi = 0.7978845608F;
        const float inner = root_two_over_pi *
                            (value + coefficient * value * value * value);
        const float tanh_inner = std::tanh(inner);
        const float derivative_inner = root_two_over_pi *
                                       (1.0F + 3.0F * coefficient * value * value);
        return 0.5F * (1.0F + tanh_inner) +
               0.5F * value * (1.0F - tanh_inner * tanh_inner) * derivative_inner;
    }
};

// These compile-time switches describe the neuron rather than hiding its
// behavior behind a runtime flag. Disabled paths remove their weight banks.
template <bool Recurrent = true,
          bool MessageFeedback = true,
          bool Gated = true,
          bool Normalize = true,
          bool Residual = true,
          bool EligibilityTrace = true>
struct FeatureSet {
    static constexpr bool recurrent = Recurrent;
    static constexpr bool message_feedback = MessageFeedback;
    static constexpr bool gated = Gated;
    static constexpr bool normalize = Normalize;
    static constexpr bool residual = Residual;
    static constexpr bool eligibility_trace = EligibilityTrace;
};

using BasicFeatures = FeatureSet<false, false, false, false, false, false>;
using RecurrentFeatures = FeatureSet<true, false, false, false, true, true>;
using ModernFeatures = FeatureSet<true, true, true, true, true, true>;

template <bool Enabled, std::size_t Count, template <std::size_t> class Storage>
struct OptionalStorage {};

template <std::size_t Count, template <std::size_t> class Storage>
struct OptionalStorage<true, Count, Storage> {
    Storage<Count> values;
};

template <std::size_t InputWidth,
          std::size_t StateWidth = InputWidth,
          std::size_t OutputWidth = InputWidth,
          typename Activation = SiluActivation,
          template <std::size_t> class WeightStorage = DenseStorage,
          template <std::size_t> class StateStorage = DenseStorage,
          typename Features = ModernFeatures>
class AdaptiveNeuron {
    static_assert(InputWidth > 0U && StateWidth > 0U && OutputWidth > 0U,
                  "neuron dimensions must be non-zero");

    using InputWeights = WeightStorage<InputWidth * OutputWidth>;
    using RecurrentWeights = WeightStorage<StateWidth * OutputWidth>;
    using MessageWeights = WeightStorage<StateWidth * OutputWidth>;
    using GateInputWeights = WeightStorage<InputWidth * OutputWidth>;
    using GateRecurrentWeights = WeightStorage<StateWidth * OutputWidth>;

public:
    using input_type = std::array<float, InputWidth>;
    using output_type = std::array<float, OutputWidth>;
    using state_type = std::array<float, StateWidth>;

    static constexpr std::size_t input_width = InputWidth;
    static constexpr std::size_t state_width = StateWidth;
    static constexpr std::size_t output_width = OutputWidth;
    static constexpr std::size_t parameter_count =
        InputWidth * OutputWidth + OutputWidth +
        (Features::recurrent ? StateWidth * OutputWidth : 0U) +
        (Features::message_feedback ? StateWidth * OutputWidth : 0U) +
        (Features::gated ? InputWidth * OutputWidth + StateWidth * OutputWidth + OutputWidth : 0U);

    explicit AdaptiveNeuron(std::uint64_t seed = 0xA17E'BEEF'1234'5678ULL) {
        initialize(seed);
    }

    [[nodiscard]] output_type forward(std::span<const float> input,
                                      std::span<const float> message = {}) {
        validate_input(input, message);
        output_type candidate{};
        for (std::size_t output = 0U; output < OutputWidth; ++output) {
            float pre_activation = bias_.get(output);
            for (std::size_t feature = 0U; feature < InputWidth; ++feature) {
                pre_activation += input_weights_.get(output * InputWidth + feature) * input[feature];
            }
            if constexpr (Features::recurrent) {
                for (std::size_t state = 0U; state < StateWidth; ++state) {
                    pre_activation += recurrent_weights_.values.get(output * StateWidth + state) *
                                      state_.get(state);
                }
            }
            if constexpr (Features::message_feedback) {
                for (std::size_t state = 0U; state < StateWidth; ++state) {
                    pre_activation += message_weights_.values.get(output * StateWidth + state) *
                                      (message.empty() ? 0.0F : message[state]);
                }
            }
            const float activated = Activation::apply(pre_activation);
            float result = activated;
            if constexpr (Features::gated) {
                float gate_pre_activation = gate_bias_.values.get(output);
                for (std::size_t feature = 0U; feature < InputWidth; ++feature) {
                    gate_pre_activation += gate_input_weights_.values.get(
                        output * InputWidth + feature) * input[feature];
                }
                for (std::size_t state = 0U; state < StateWidth; ++state) {
                    gate_pre_activation += gate_recurrent_weights_.values.get(
                        output * StateWidth + state) * state_.get(state);
                }
                const float gate = SigmoidActivation::apply(gate_pre_activation);
                const float previous = output_[output];
                result = gate * activated + (1.0F - gate) * previous;
            } else if constexpr (Features::residual) {
                result += 0.10F * output_[output];
            }
            candidate[output] = result;
        }

        if constexpr (Features::normalize) {
            normalize(candidate);
        }
        output_ = candidate;
        for (std::size_t state = 0U; state < StateWidth; ++state) {
            const float next = output_[state % OutputWidth];
            const float old = state_.get(state);
            state_.set(state, 0.90F * old + 0.10F * next);
            if constexpr (Features::eligibility_trace) {
                trace_.values.add(state, 0.95F * trace_.values.get(state) + 0.05F * next);
            }
        }
        return output_;
    }

    [[nodiscard]] output_type learn(std::span<const float> input,
                                    std::span<const float> target,
                                    float learning_rate,
                                    std::span<const float> message = {}) {
        if (target.size() != OutputWidth) {
            throw std::invalid_argument("neuron target width does not match output width");
        }
        if (!(learning_rate >= 0.0F) || !std::isfinite(learning_rate)) {
            throw std::invalid_argument("neuron learning rate must be finite and non-negative");
        }
        const output_type prediction = forward(input, message);
        for (std::size_t output = 0U; output < OutputWidth; ++output) {
            const float error = (target[output] - prediction[output]) *
                                Activation::derivative(prediction[output]);
            bias_.add(output, learning_rate * error);
            for (std::size_t feature = 0U; feature < InputWidth; ++feature) {
                input_weights_.add(output * InputWidth + feature,
                                   learning_rate * error * input[feature]);
            }
            if constexpr (Features::recurrent) {
                for (std::size_t state = 0U; state < StateWidth; ++state) {
                    recurrent_weights_.values.add(output * StateWidth + state,
                                                  learning_rate * error * state_.get(state));
                }
            }
            if constexpr (Features::message_feedback) {
                for (std::size_t state = 0U; state < StateWidth; ++state) {
                    const float message_value = message.empty() ? 0.0F : message[state];
                    message_weights_.values.add(output * StateWidth + state,
                                                learning_rate * error * message_value);
                }
            }
        }
        return prediction;
    }

    void reset_state() noexcept {
        for (std::size_t index = 0U; index < StateWidth; ++index) {
            state_.set(index, 0.0F);
            if constexpr (Features::eligibility_trace) {
                trace_.values.set(index, 0.0F);
            }
        }
        output_.fill(0.0F);
    }

    [[nodiscard]] const output_type& output() const noexcept { return output_; }

    [[nodiscard]] state_type state() const noexcept {
        state_type result{};
        for (std::size_t index = 0U; index < StateWidth; ++index) {
            result[index] = state_.get(index);
        }
        return result;
    }

    [[nodiscard]] state_type eligibility_trace() const noexcept {
        state_type result{};
        if constexpr (Features::eligibility_trace) {
            for (std::size_t index = 0U; index < StateWidth; ++index) {
                result[index] = trace_.values.get(index);
            }
        }
        return result;
    }

    [[nodiscard]] static constexpr std::size_t bytes_per_instance() noexcept {
        return sizeof(AdaptiveNeuron);
    }

private:
    static std::uint64_t mix(std::uint64_t value) noexcept {
        value += 0x9E37'79B9'7F4A'7C15ULL;
        value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
        return value ^ (value >> 31U);
    }

    template <typename Storage>
    static void initialize_storage(Storage& storage, std::uint64_t seed,
                                   float scale = 0.05F) noexcept {
        for (std::size_t index = 0U; index < Storage::element_count; ++index) {
            const std::uint64_t bits = mix(seed + index * 0x9E37ULL);
            const float unit = static_cast<float>(bits & 0xFFFFU) / 65535.0F;
            storage.set(index, (unit * 2.0F - 1.0F) * scale);
        }
    }

    void initialize(std::uint64_t seed) noexcept {
        initialize_storage(input_weights_, seed + 1U);
        initialize_storage(bias_, seed + 2U, 0.01F);
        if constexpr (Features::recurrent) {
            initialize_storage(recurrent_weights_.values, seed + 3U);
        }
        if constexpr (Features::message_feedback) {
            initialize_storage(message_weights_.values, seed + 4U);
        }
        if constexpr (Features::gated) {
            initialize_storage(gate_input_weights_.values, seed + 5U);
            initialize_storage(gate_recurrent_weights_.values, seed + 6U);
            initialize_storage(gate_bias_.values, seed + 7U, 0.01F);
        }
        reset_state();
    }

    static void validate_input(std::span<const float> input,
                               std::span<const float> message) {
        if (input.size() != InputWidth) {
            throw std::invalid_argument("neuron input width does not match input width");
        }
        if constexpr (Features::message_feedback) {
            if (!message.empty() && message.size() != StateWidth) {
                throw std::invalid_argument("neuron feedback width does not match state width");
            }
        }
    }

    static void normalize(output_type& values) noexcept {
        float mean = 0.0F;
        for (const float value : values) {
            mean += value;
        }
        mean /= static_cast<float>(OutputWidth);
        float variance = 0.0F;
        for (const float value : values) {
            const float centered = value - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(OutputWidth);
        const float inverse = 1.0F / std::sqrt(variance + 1.0e-5F);
        for (float& value : values) {
            value = (value - mean) * inverse;
        }
    }

    InputWeights input_weights_;
    WeightStorage<OutputWidth> bias_;
    OptionalStorage<Features::recurrent, OutputWidth * StateWidth, WeightStorage>
        recurrent_weights_;
    OptionalStorage<Features::message_feedback, OutputWidth * StateWidth, WeightStorage>
        message_weights_;
    OptionalStorage<Features::gated, OutputWidth * InputWidth, WeightStorage>
        gate_input_weights_;
    OptionalStorage<Features::gated, OutputWidth * StateWidth, WeightStorage>
        gate_recurrent_weights_;
    OptionalStorage<Features::gated, OutputWidth, WeightStorage> gate_bias_;
    StateStorage<StateWidth> state_;
    OptionalStorage<Features::eligibility_trace, StateWidth, StateStorage> trace_;
    output_type output_{};
};

template <std::size_t Width, std::size_t StateWidth = Width,
          std::size_t OutputWidth = Width>
using BasicNeuron = AdaptiveNeuron<Width, StateWidth, OutputWidth,
                                   ReluActivation, DenseStorage, DenseStorage,
                                   BasicFeatures>;

template <std::size_t Width, std::size_t StateWidth = Width,
          std::size_t OutputWidth = Width>
using ModernGatedNeuron = AdaptiveNeuron<Width, StateWidth, OutputWidth,
                                         SiluActivation, DenseStorage, DenseStorage,
                                         ModernFeatures>;

template <std::size_t Width, std::size_t StateWidth = Width,
          std::size_t OutputWidth = Width>
using CompactModernNeuron = AdaptiveNeuron<Width, StateWidth, OutputWidth,
                                           GeluActivation, QuantizedInt8Storage,
                                           DenseStorage, ModernFeatures>;

template <std::size_t Width, std::size_t StateWidth = Width,
          std::size_t OutputWidth = Width>
using BinaryInferenceNeuron = AdaptiveNeuron<Width, StateWidth, OutputWidth,
                                             ReluActivation, BinaryStorage,
                                             DenseStorage, BasicFeatures>;

// Sparse mixture-of-experts routing is an optional higher-capacity preset.
// Only TopK experts update their state for each input, so different neuron
// families can coexist behind one fixed-width interface without paying for
// every expert on every token.
template <std::size_t InputWidth,
          std::size_t StateWidth = InputWidth,
          std::size_t OutputWidth = InputWidth,
          std::size_t ExpertCount = 4U,
          std::size_t TopK = 2U,
          typename Activation = SiluActivation,
          template <std::size_t> class WeightStorage = DenseStorage,
          template <std::size_t> class StateStorage = DenseStorage,
          typename Features = ModernFeatures>
class SparseMixtureNeuron {
    static_assert(ExpertCount > 0U && TopK > 0U && TopK <= ExpertCount,
                  "mixture neurons require 1..ExpertCount active experts");
    using Expert = AdaptiveNeuron<InputWidth, StateWidth, OutputWidth,
                                  Activation, WeightStorage, StateStorage, Features>;

public:
    using output_type = typename Expert::output_type;
    static constexpr std::size_t parameter_count =
        ExpertCount * Expert::parameter_count + ExpertCount * (InputWidth + 1U);

    explicit SparseMixtureNeuron(std::uint64_t seed = 0x5A'5EED'1234'5678ULL) {
        for (std::size_t expert = 0U; expert < ExpertCount; ++expert) {
            experts_[expert] = Expert(mix(seed + expert * 0x9E37ULL));
        }
        initialize_router(seed);
    }

    [[nodiscard]] output_type forward(std::span<const float> input,
                                      std::span<const float> message = {}) {
        if (input.size() != InputWidth) {
            throw std::invalid_argument("mixture neuron input width does not match input width");
        }
        std::array<float, ExpertCount> scores{};
        std::array<std::size_t, ExpertCount> order{};
        for (std::size_t expert = 0U; expert < ExpertCount; ++expert) {
            order[expert] = expert;
            scores[expert] = router_bias_.get(expert);
            for (std::size_t feature = 0U; feature < InputWidth; ++feature) {
                scores[expert] += router_weights_.get(expert * InputWidth + feature) *
                                  input[feature];
            }
        }
        std::partial_sort(order.begin(), order.begin() + TopK, order.end(),
                          [&scores](std::size_t left, std::size_t right) {
                              return scores[left] > scores[right];
                          });
        const float maximum = scores[order[0U]];
        float denominator = 0.0F;
        for (std::size_t active = 0U; active < TopK; ++active) {
            routing_weights_[active] = std::exp(scores[order[active]] - maximum);
            denominator += routing_weights_[active];
        }

        output_type result{};
        for (std::size_t active = 0U; active < TopK; ++active) {
            routing_weights_[active] /= denominator;
            active_experts_[active] = order[active];
            const output_type expert_output = experts_[order[active]].forward(input, message);
            for (std::size_t output = 0U; output < OutputWidth; ++output) {
                result[output] += routing_weights_[active] * expert_output[output];
            }
        }
        output_ = result;
        return output_;
    }

    [[nodiscard]] output_type learn(std::span<const float> input,
                                    std::span<const float> target,
                                    float learning_rate,
                                    std::span<const float> message = {}) {
        const output_type prediction = forward(input, message);
        for (std::size_t active = 0U; active < TopK; ++active) {
            experts_[active_experts_[active]].learn(input, target,
                                                    learning_rate * routing_weights_[active],
                                                    message);
        }
        return prediction;
    }

    void reset_state() noexcept {
        for (Expert& expert : experts_) {
            expert.reset_state();
        }
        output_.fill(0.0F);
        routing_weights_.fill(0.0F);
        active_experts_.fill(0U);
    }

    [[nodiscard]] const output_type& output() const noexcept { return output_; }
    [[nodiscard]] const std::array<float, TopK>& routing_weights() const noexcept {
        return routing_weights_;
    }
    [[nodiscard]] const std::array<std::size_t, TopK>& active_experts() const noexcept {
        return active_experts_;
    }
    [[nodiscard]] static constexpr std::size_t bytes_per_instance() noexcept {
        return sizeof(SparseMixtureNeuron);
    }

private:
    static std::uint64_t mix(std::uint64_t value) noexcept {
        value += 0x9E37'79B9'7F4A'7C15ULL;
        value = (value ^ (value >> 30U)) * 0xBF58'476D'1CE4'E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D0'49BB'1331'11EBULL;
        return value ^ (value >> 31U);
    }

    void initialize_router(std::uint64_t seed) noexcept {
        for (std::size_t index = 0U; index < ExpertCount * InputWidth; ++index) {
            const float unit = static_cast<float>(mix(seed + index) & 0xFFFFU) / 65535.0F;
            router_weights_.set(index, (unit * 2.0F - 1.0F) * 0.05F);
        }
        for (std::size_t index = 0U; index < ExpertCount; ++index) {
            router_bias_.set(index, 0.0F);
        }
    }

    std::array<Expert, ExpertCount> experts_;
    WeightStorage<ExpertCount * InputWidth> router_weights_;
    WeightStorage<ExpertCount> router_bias_;
    std::array<float, TopK> routing_weights_{};
    std::array<std::size_t, TopK> active_experts_{};
    output_type output_{};
};

template <std::size_t Width, std::size_t ExpertCount = 4U, std::size_t TopK = 2U>
using SparseModernNeuron = SparseMixtureNeuron<Width, Width, Width, ExpertCount, TopK>;

}  // namespace agentari::neuron
