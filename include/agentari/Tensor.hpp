#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace agentari::gpu {
class VulkanComputeBackend;
}

namespace agentari::nn {

using Shape = std::vector<std::size_t>;

enum class TensorBackend : std::uint8_t {
    cpu,
    vulkan,
};

// The backend pointer is non-owning and must outlive all tensor operations
// performed while Vulkan is selected. CPU remains the safe default.
void set_tensor_backend(TensorBackend backend,
                        gpu::VulkanComputeBackend* vulkan_backend = nullptr) noexcept;
[[nodiscard]] TensorBackend tensor_backend() noexcept;

namespace detail {

struct AutogradNode;

struct TensorImpl {
    Shape shape;
    std::vector<float> data;
    std::vector<float> grad;
    bool requires_grad{false};
    std::shared_ptr<AutogradNode> creator;
};

struct AutogradNode {
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::weak_ptr<TensorImpl> output;
    std::function<void()> backward;
};

}  // namespace detail

// A small owning tensor handle. Copies share storage, which lets autograd
// accumulate gradients into the original parameter or input tensor.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(Shape shape, float value = 0.0F, bool requires_grad = false);
    Tensor(Shape shape, std::vector<float> values, bool requires_grad = false);

    [[nodiscard]] static Tensor scalar(float value, bool requires_grad = false);
    [[nodiscard]] static Tensor zeros(Shape shape, bool requires_grad = false);
    [[nodiscard]] static Tensor ones(Shape shape, bool requires_grad = false);
    [[nodiscard]] static Tensor random_normal(Shape shape,
                                               float mean,
                                               float standard_deviation,
                                               std::uint64_t seed,
                                               bool requires_grad = false);

    [[nodiscard]] bool defined() const noexcept;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] std::size_t ndim() const;
    [[nodiscard]] std::size_t numel() const;
    [[nodiscard]] bool requires_grad() const;
    [[nodiscard]] bool has_grad() const;

    [[nodiscard]] const std::vector<float>& values() const;
    [[nodiscard]] std::vector<float>& mutable_values();
    [[nodiscard]] const std::vector<float>& grad_values() const;

    [[nodiscard]] float at(std::initializer_list<std::size_t> indexes) const;
    float& at(std::initializer_list<std::size_t> indexes);

    void set_requires_grad(bool enabled);
    void zero_grad();
    void backward() const;
    void backward(const Tensor& gradient) const;

    [[nodiscard]] Tensor detach() const;
    [[nodiscard]] Tensor reshape(Shape new_shape) const;
    [[nodiscard]] Tensor transpose() const;
    [[nodiscard]] std::string describe() const;

    // Internal bridge used by the neural-operation implementation. Keeping
    // storage behind a shared implementation gives tensors value-like syntax
    // without copying the graph or gradient buffers.
    [[nodiscard]] const std::shared_ptr<detail::TensorImpl>& internal() const noexcept;

private:
    explicit Tensor(std::shared_ptr<detail::TensorImpl> implementation);

    std::shared_ptr<detail::TensorImpl> implementation_;

    friend Tensor operator+(const Tensor&, const Tensor&);
    friend Tensor operator-(const Tensor&, const Tensor&);
    friend Tensor operator*(const Tensor&, const Tensor&);
    friend Tensor operator*(const Tensor&, float);
    friend Tensor operator*(float, const Tensor&);
    friend Tensor operator/(const Tensor&, float);
    friend Tensor matmul(const Tensor&, const Tensor&);
    friend Tensor relu(const Tensor&);
    friend Tensor silu(const Tensor&);
    friend Tensor gelu(const Tensor&);
    friend Tensor sigmoid(const Tensor&);
    friend Tensor sum(const Tensor&);
    friend Tensor mean(const Tensor&);
    friend Tensor softmax(const Tensor&);
    friend Tensor cross_entropy(const Tensor&, const std::vector<std::size_t>&);
    friend Tensor embedding(const Tensor&, const std::vector<std::size_t>&);
    friend Tensor rms_norm(const Tensor&, const Tensor&, float);
    friend Tensor apply_rope(const Tensor&, std::size_t, float, std::size_t);
    friend Tensor scaled_dot_product_attention(const Tensor&, const Tensor&, const Tensor&,
                                               bool, std::size_t);
    friend Tensor concatenate_sequence(const Tensor&, const Tensor&);
};

class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool previous_state_{true};
};

[[nodiscard]] bool gradients_enabled() noexcept;

[[nodiscard]] Tensor operator+(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor operator-(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor operator*(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor operator*(const Tensor& value, float scalar);
[[nodiscard]] Tensor operator*(float scalar, const Tensor& value);
[[nodiscard]] Tensor operator/(const Tensor& value, float scalar);

[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right);
[[nodiscard]] Tensor relu(const Tensor& value);
[[nodiscard]] Tensor silu(const Tensor& value);
[[nodiscard]] Tensor gelu(const Tensor& value);
[[nodiscard]] Tensor sigmoid(const Tensor& value);
[[nodiscard]] Tensor sum(const Tensor& value);
[[nodiscard]] Tensor mean(const Tensor& value);
[[nodiscard]] Tensor softmax(const Tensor& value);
[[nodiscard]] Tensor cross_entropy(const Tensor& logits,
                                   const std::vector<std::size_t>& targets);
[[nodiscard]] Tensor embedding(const Tensor& weights,
                               const std::vector<std::size_t>& token_ids);
[[nodiscard]] Tensor rms_norm(const Tensor& value, const Tensor& weight, float epsilon);

// Rotates pairs of features in the final dimension. The position offset is
// used by incremental decoding so cached keys and new queries share positions.
[[nodiscard]] Tensor apply_rope(const Tensor& value,
                                std::size_t rotary_dimension,
                                float theta,
                                std::size_t position_offset = 0U);

// q: [batch, query_heads, query_length, head_dimension]
// k: [batch, key_value_heads, key_length, head_dimension]
// v: [batch, key_value_heads, key_length, value_dimension]
// query_heads must be divisible by key_value_heads. Causal masking is applied
// against key positions beginning at query_position_offset.
[[nodiscard]] Tensor scaled_dot_product_attention(const Tensor& query,
                                                  const Tensor& key,
                                                  const Tensor& value,
                                                  bool causal,
                                                  std::size_t query_position_offset = 0U);

// Concatenates two [batch, heads, sequence, features] tensors along sequence.
// This is intentionally inference-oriented and does not build an autograd
// node; training uses one complete sequence at a time.
[[nodiscard]] Tensor concatenate_sequence(const Tensor& prefix, const Tensor& suffix);

// Converts a row-major [sequence, heads * features] projection into the
// attention layout [1, heads, sequence, features] without reinterpreting the
// row-major projection storage. The inverse produces [sequence, heads * features].
[[nodiscard]] Tensor split_attention_heads(const Tensor& value, std::size_t head_count);
[[nodiscard]] Tensor merge_attention_heads(const Tensor& value);

[[nodiscard]] std::size_t argmax_last_row(const Tensor& value);
[[nodiscard]] float l2_norm(const std::vector<Tensor*>& parameters);
void clip_grad_norm(const std::vector<Tensor*>& parameters, float maximum_norm);

}  // namespace agentari::nn
