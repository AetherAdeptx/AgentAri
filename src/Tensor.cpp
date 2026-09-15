#include "agentari/Tensor.hpp"
#include "agentari/Kernel.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/System.hpp"
#include "agentari/VulkanBackend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace agentari::nn {
namespace {

thread_local bool gradients_enabled_state = true;
std::atomic<TensorBackend> selected_tensor_backend{TensorBackend::cpu};
std::atomic<gpu::VulkanComputeBackend*> selected_vulkan_backend{nullptr};

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
float dot_product_avx2_fma(const float* left, const float* right, std::size_t count) noexcept {
    __m256 accumulator = _mm256_setzero_ps();
    std::size_t index = 0U;
    for (; index + 8U <= count; index += 8U) {
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(left + index),
                                      _mm256_loadu_ps(right + index), accumulator);
    }
    alignas(32) float lanes[8]{};
    _mm256_store_ps(lanes, accumulator);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; index < count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}
#endif
#endif

float dot_product(const float* left, const float* right, std::size_t count) noexcept {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    if (agentari::system::cpu_capabilities().avx2 &&
        agentari::system::cpu_capabilities().fma) {
        return dot_product_avx2_fma(left, right, count);
    }
#endif
#endif
    float result = 0.0F;
    for (std::size_t index = 0U; index < count; ++index) {
        result += left[index] * right[index];
    }
    return result;
}

std::size_t element_count(const Shape& shape) {
    if (shape.empty()) {
        return 1U;
    }

    std::size_t count = 1U;
    for (const std::size_t dimension : shape) {
        if (dimension != 0U && count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("tensor shape is too large");
        }
        count *= dimension;
    }
    return count;
}

void require_defined(const Tensor& value, const char* operation) {
    if (!value.defined()) {
        throw std::invalid_argument(std::string(operation) + " received an undefined tensor");
    }
}

std::vector<std::size_t> strides_for(const Shape& shape) {
    std::vector<std::size_t> strides(shape.size(), 1U);
    std::size_t stride = 1U;
    for (std::size_t index = shape.size(); index > 0U; --index) {
        strides[index - 1U] = stride;
        stride *= shape[index - 1U];
    }
    return strides;
}

std::size_t offset_for(const Shape& shape, std::initializer_list<std::size_t> indexes) {
    if (indexes.size() != shape.size()) {
        throw std::invalid_argument("tensor index rank does not match tensor rank");
    }

    const auto strides = strides_for(shape);
    std::size_t offset = 0U;
    std::size_t dimension_index = 0U;
    for (const std::size_t index : indexes) {
        if (index >= shape[dimension_index]) {
            throw std::out_of_range("tensor index is outside its shape");
        }
        offset += index * strides[dimension_index];
        ++dimension_index;
    }
    return offset;
}

bool should_track(const Tensor& value) {
    return gradients_enabled() && value.requires_grad();
}

bool should_track(const Tensor& left, const Tensor& right) {
    return gradients_enabled() && (left.requires_grad() || right.requires_grad());
}

void attach_node(Tensor& output,
                 std::vector<std::shared_ptr<detail::TensorImpl>> parents,
                 std::function<void()> backward) {
    if (!output.requires_grad()) {
        return;
    }

    auto node = std::make_shared<detail::AutogradNode>();
    node->parents = std::move(parents);
    node->output = output.internal();
    node->backward = std::move(backward);
    output.internal()->creator = std::move(node);
}

bool can_broadcast(const Tensor& source, const Shape& destination_shape) {
    if (source.shape() == destination_shape || source.numel() == 1U) {
        return true;
    }
    return source.ndim() == 1U && !destination_shape.empty() &&
           source.shape()[0] == destination_shape.back();
}

std::size_t broadcast_index(const Tensor& source,
                            std::size_t destination_index,
                            const Shape& destination_shape) {
    if (source.numel() == 1U) {
        return 0U;
    }
    if (source.shape() == destination_shape) {
        return destination_index;
    }
    return destination_index % source.shape()[0];
}

void require_binary_shapes(const Tensor& left, const Tensor& right, const char* operation) {
    require_defined(left, operation);
    require_defined(right, operation);
    const Shape destination_shape = left.numel() == 1U ? right.shape() : left.shape();
    if (!can_broadcast(left, destination_shape) || !can_broadcast(right, destination_shape)) {
        throw std::invalid_argument(std::string(operation) +
                                    " supports equal shapes, scalar broadcasting, or final-dimension broadcasting");
    }
}

float broadcast_value(const Tensor& value, std::size_t index, const Shape& destination_shape) {
    return value.values()[broadcast_index(value, index, destination_shape)];
}

Tensor binary_elementwise(const Tensor& left,
                          const Tensor& right,
                          const char* operation,
                          const std::function<float(float, float)>& forward,
                          const std::function<float(float, float, float)>& left_derivative,
                          const std::function<float(float, float, float)>& right_derivative,
                          gpu::ElementwiseOperation gpu_operation) {
    require_binary_shapes(left, right, operation);
    const Shape output_shape = left.numel() == 1U ? right.shape() : left.shape();
    std::vector<float> output_data(element_count(output_shape), 0.0F);
    bool used_vulkan = false;
    if (left.shape() == right.shape() && tensor_backend() == TensorBackend::vulkan) {
        if (gpu::VulkanComputeBackend* backend =
                selected_vulkan_backend.load(std::memory_order_acquire);
            backend != nullptr) {
            used_vulkan = backend->elementwise(gpu_operation, left.values().data(),
                                               right.values().data(), output_data.size(),
                                               output_data.data());
        }
    }
    if (!used_vulkan) {
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, output_data.size(), [&](std::size_t index) {
            output_data[index] = forward(broadcast_value(left, index, output_shape),
                                         broadcast_value(right, index, output_shape));
        }, parallel_config);
    }

    Tensor output(output_shape, std::move(output_data), should_track(left, right));
    const auto left_impl = left.internal();
    const auto right_impl = right.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {left_impl, right_impl},
                [left_impl, right_impl, output_impl, left_derivative, right_derivative]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr) {
                        return;
                    }
                    const std::size_t output_count = result->data.size();
                    const auto update_index = [&](std::size_t index) {
                        const float output_gradient = result->grad[index];
                        const std::size_t left_index = left_impl->data.size() == 1U
                                                           ? 0U
                                                           : (left_impl->shape == result->shape
                                                                  ? index
                                                                  : index % left_impl->shape[0]);
                        const std::size_t right_index = right_impl->data.size() == 1U
                                                            ? 0U
                                                            : (right_impl->shape == result->shape
                                                                   ? index
                                                                   : index % right_impl->shape[0]);
                        const float left_value = left_impl->data[left_index];
                        const float right_value = right_impl->data[right_index];
                        if (left_impl->requires_grad) {
                            const float amount = left_derivative(output_gradient, left_value, right_value);
                            left_impl->grad[left_index] += amount;
                        }
                        if (right_impl->requires_grad) {
                            const float amount = right_derivative(output_gradient, left_value, right_value);
                            right_impl->grad[right_index] += amount;
                        }
                    };
                    if (left_impl->shape == result->shape && right_impl->shape == result->shape) {
                        parallel::Config parallel_config;
                        parallel_config.minimum_parallel_work = 2048U;
                        parallel::parallel_for(0U, output_count, update_index, parallel_config);
                    } else {
                        for (std::size_t index = 0U; index < output_count; ++index) {
                            update_index(index);
                        }
                    }
                });
    return output;
}

Tensor unary_elementwise(const Tensor& value,
                         const char* operation,
                         const std::function<float(float)>& forward,
                         const std::function<float(float)>& derivative,
                         gpu::ElementwiseOperation gpu_operation) {
    require_defined(value, operation);
    std::vector<float> output_data(value.numel(), 0.0F);
    bool used_vulkan = false;
    if (tensor_backend() == TensorBackend::vulkan) {
        if (gpu::VulkanComputeBackend* backend =
                selected_vulkan_backend.load(std::memory_order_acquire);
            backend != nullptr) {
            used_vulkan = backend->elementwise(gpu_operation, value.values().data(), nullptr,
                                               output_data.size(), output_data.data());
        }
    }
    if (!used_vulkan) {
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, value.numel(), [&](std::size_t index) {
            output_data[index] = forward(value.values()[index]);
        }, parallel_config);
    }

    Tensor output(value.shape(), std::move(output_data), should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, derivative]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, result->data.size(), [&](std::size_t index) {
            value_impl->grad[index] += result->grad[index] * derivative(value_impl->data[index]);
        }, parallel_config);
    });
    return output;
}

}  // namespace

void set_tensor_backend(TensorBackend backend,
                        gpu::VulkanComputeBackend* vulkan_backend) noexcept {
    selected_vulkan_backend.store(vulkan_backend, std::memory_order_release);
    selected_tensor_backend.store(backend, std::memory_order_release);
}

TensorBackend tensor_backend() noexcept {
    return selected_tensor_backend.load(std::memory_order_acquire);
}

NoGradGuard::NoGradGuard()
    : previous_state_(gradients_enabled_state) {
    gradients_enabled_state = false;
}

NoGradGuard::~NoGradGuard() {
    gradients_enabled_state = previous_state_;
}

bool gradients_enabled() noexcept {
    return gradients_enabled_state;
}

Tensor::Tensor(Shape shape, float value, bool requires_grad)
    : implementation_(std::make_shared<detail::TensorImpl>()) {
    implementation_->shape = std::move(shape);
    implementation_->data.assign(element_count(implementation_->shape), value);
    implementation_->requires_grad = requires_grad;
    if (requires_grad) {
        implementation_->grad.assign(implementation_->data.size(), 0.0F);
    }
}

Tensor::Tensor(Shape shape, std::vector<float> values, bool requires_grad)
    : implementation_(std::make_shared<detail::TensorImpl>()) {
    implementation_->shape = std::move(shape);
    const std::size_t expected_count = element_count(implementation_->shape);
    if (values.size() != expected_count) {
        throw std::invalid_argument("tensor data count does not match tensor shape");
    }
    implementation_->data = std::move(values);
    implementation_->requires_grad = requires_grad;
    if (requires_grad) {
        implementation_->grad.assign(implementation_->data.size(), 0.0F);
    }
}

Tensor::Tensor(std::shared_ptr<detail::TensorImpl> implementation)
    : implementation_(std::move(implementation)) {}

Tensor Tensor::scalar(float value, bool requires_grad) {
    return Tensor(Shape{}, std::vector<float>{value}, requires_grad);
}

Tensor Tensor::zeros(Shape shape, bool requires_grad) {
    return Tensor(std::move(shape), 0.0F, requires_grad);
}

Tensor Tensor::ones(Shape shape, bool requires_grad) {
    return Tensor(std::move(shape), 1.0F, requires_grad);
}

Tensor Tensor::random_normal(Shape shape,
                             float mean_value,
                             float standard_deviation,
                             std::uint64_t seed,
                             bool requires_grad) {
    if (!(standard_deviation >= 0.0F)) {
        throw std::invalid_argument("normal distribution standard deviation must be non-negative");
    }
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> distribution(mean_value, standard_deviation);
    const std::size_t count = element_count(shape);
    std::vector<float> values(count, 0.0F);
    for (float& value : values) {
        value = distribution(generator);
    }
    return Tensor(std::move(shape), std::move(values), requires_grad);
}

bool Tensor::defined() const noexcept {
    return implementation_ != nullptr;
}

const Shape& Tensor::shape() const {
    require_defined(*this, "shape");
    return implementation_->shape;
}

std::size_t Tensor::ndim() const {
    return shape().size();
}

std::size_t Tensor::numel() const {
    require_defined(*this, "numel");
    return implementation_->data.size();
}

bool Tensor::requires_grad() const {
    return defined() && implementation_->requires_grad;
}

bool Tensor::has_grad() const {
    if (!requires_grad()) {
        return false;
    }
    return std::any_of(implementation_->grad.begin(), implementation_->grad.end(),
                       [](float value) { return value != 0.0F; });
}

const std::vector<float>& Tensor::values() const {
    require_defined(*this, "values");
    return implementation_->data;
}

std::vector<float>& Tensor::mutable_values() {
    require_defined(*this, "mutable_values");
    return implementation_->data;
}

const std::vector<float>& Tensor::grad_values() const {
    require_defined(*this, "grad_values");
    return implementation_->grad;
}

float Tensor::at(std::initializer_list<std::size_t> indexes) const {
    return values()[offset_for(shape(), indexes)];
}

float& Tensor::at(std::initializer_list<std::size_t> indexes) {
    return mutable_values()[offset_for(shape(), indexes)];
}

void Tensor::set_requires_grad(bool enabled) {
    require_defined(*this, "set_requires_grad");
    implementation_->requires_grad = enabled;
    implementation_->creator.reset();
    if (enabled) {
        implementation_->grad.assign(implementation_->data.size(), 0.0F);
    } else {
        implementation_->grad.clear();
    }
}

void Tensor::zero_grad() {
    if (requires_grad()) {
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, implementation_->grad.size(),
                               [this](std::size_t index) {
                                   implementation_->grad[index] = 0.0F;
                               }, parallel_config);
    }
}

void Tensor::backward() const {
    require_defined(*this, "backward");
    if (numel() != 1U) {
        throw std::invalid_argument("backward() requires a scalar tensor; provide an explicit gradient otherwise");
    }
    backward(Tensor::scalar(1.0F));
}

void Tensor::backward(const Tensor& gradient) const {
    require_defined(*this, "backward");
    require_defined(gradient, "backward");
    if (!requires_grad()) {
        throw std::invalid_argument("cannot backpropagate from a tensor that does not require gradients");
    }
    if (shape() != gradient.shape()) {
        throw std::invalid_argument("backward gradient shape does not match output shape");
    }

    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 2048U;
    parallel::parallel_for(0U, numel(), [&](std::size_t index) {
        implementation_->grad[index] += gradient.values()[index];
    }, parallel_config);

    std::vector<std::shared_ptr<detail::TensorImpl>> topology;
    std::unordered_set<const detail::TensorImpl*> visited;
    const std::function<void(const std::shared_ptr<detail::TensorImpl>&)> visit =
        [&topology, &visited, &visit](const std::shared_ptr<detail::TensorImpl>& value) {
            if (value == nullptr || !visited.insert(value.get()).second) {
                return;
            }
            if (value->creator != nullptr) {
                for (const auto& parent : value->creator->parents) {
                    visit(parent);
                }
            }
            topology.push_back(value);
        };
    visit(implementation_);

    for (auto iterator = topology.rbegin(); iterator != topology.rend(); ++iterator) {
        if ((*iterator)->creator != nullptr && (*iterator)->creator->backward) {
            (*iterator)->creator->backward();
        }
    }
}

Tensor Tensor::detach() const {
    require_defined(*this, "detach");
    return Tensor(shape(), values(), false);
}

Tensor Tensor::reshape(Shape new_shape) const {
    require_defined(*this, "reshape");
    if (element_count(new_shape) != numel()) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    Tensor output(std::move(new_shape), values(), should_track(*this));
    const auto value_impl = implementation_;
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, result->grad.size(), [&](std::size_t index) {
            value_impl->grad[index] += result->grad[index];
        }, parallel_config);
    });
    return output;
}

Tensor Tensor::transpose() const {
    require_defined(*this, "transpose");
    if (ndim() != 2U) {
        throw std::invalid_argument("transpose currently requires a rank-2 tensor");
    }
    const std::size_t rows = shape()[0];
    const std::size_t columns = shape()[1];
    std::vector<float> output_data(numel(), 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 1024U;
    parallel::parallel_for(0U, rows, [&](std::size_t row) {
        for (std::size_t column = 0U; column < columns; ++column) {
            output_data[column * rows + row] = values()[row * columns + column];
        }
    }, parallel_config);

    Tensor output(Shape{columns, rows}, std::move(output_data), should_track(*this));
    const auto value_impl = implementation_;
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, rows, columns]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        parallel::Config gradient_config;
        gradient_config.minimum_parallel_work = 1024U;
        parallel::parallel_for(0U, rows, [&](std::size_t row) {
            for (std::size_t column = 0U; column < columns; ++column) {
                value_impl->grad[row * columns + column] += result->grad[column * rows + row];
            }
        }, gradient_config);
    });
    return output;
}

std::string Tensor::describe() const {
    require_defined(*this, "describe");
    std::ostringstream output;
    output << "Tensor(shape=[";
    for (std::size_t index = 0U; index < shape().size(); ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << shape()[index];
    }
    output << "], values=[";
    const std::size_t preview_count = std::min<std::size_t>(values().size(), 12U);
    output << std::setprecision(5);
    for (std::size_t index = 0U; index < preview_count; ++index) {
        if (index != 0U) {
            output << ',';
        }
        output << values()[index];
    }
    if (preview_count < values().size()) {
        output << ",...";
    }
    output << "])";
    return output.str();
}

const std::shared_ptr<detail::TensorImpl>& Tensor::internal() const noexcept {
    return implementation_;
}

Tensor operator+(const Tensor& left, const Tensor& right) {
    return binary_elementwise(left, right, "addition",
                              [](float a, float b) { return a + b; },
                              [](float gradient, float, float) { return gradient; },
                              [](float gradient, float, float) { return gradient; },
                              gpu::ElementwiseOperation::add);
}

Tensor operator-(const Tensor& left, const Tensor& right) {
    return binary_elementwise(left, right, "subtraction",
                              [](float a, float b) { return a - b; },
                              [](float gradient, float, float) { return gradient; },
                              [](float gradient, float, float) { return -gradient; },
                              gpu::ElementwiseOperation::subtract);
}

Tensor operator*(const Tensor& left, const Tensor& right) {
    return binary_elementwise(left, right, "multiplication",
                              [](float a, float b) { return a * b; },
                              [](float gradient, float, float right_value) {
                                  return gradient * right_value;
                              },
                              [](float gradient, float left_value, float) {
                                  return gradient * left_value;
                              },
                              gpu::ElementwiseOperation::multiply);
}

Tensor operator*(const Tensor& value, float scalar) {
    return value * Tensor::scalar(scalar);
}

Tensor operator*(float scalar, const Tensor& value) {
    return value * scalar;
}

Tensor operator/(const Tensor& value, float scalar) {
    if (scalar == 0.0F) {
        throw std::invalid_argument("tensor division by zero");
    }
    return value * (1.0F / scalar);
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    require_defined(left, "matmul");
    require_defined(right, "matmul");
    if (left.ndim() != 2U || right.ndim() != 2U || left.shape()[1] != right.shape()[0]) {
        throw std::invalid_argument("matmul requires rank-2 tensors with compatible inner dimensions");
    }

    const std::size_t rows = left.shape()[0];
    const std::size_t inner = left.shape()[1];
    const std::size_t columns = right.shape()[1];
    const auto checked_product = [](std::size_t first, std::size_t second) {
        if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
            throw std::overflow_error("tensor matmul shape is too large");
        }
        return first * second;
    };
    const std::size_t output_count = checked_product(rows, columns);
    const std::size_t transposed_count = checked_product(columns, inner);
    std::vector<float> output_data(output_count, 0.0F);
    bool used_vulkan = false;
    if (tensor_backend() == TensorBackend::vulkan) {
        if (gpu::VulkanComputeBackend* backend =
                selected_vulkan_backend.load(std::memory_order_acquire);
            backend != nullptr) {
            used_vulkan = backend->matmul(left.values().data(), rows, inner,
                                          right.values().data(), columns,
                                          output_data.data());
        }
    }
    if (!used_vulkan) {
        // Transposing once makes each right-hand column contiguous. That lets
        // the runtime-dispatched dot product use AVX2/FMA on CPUs that expose it.
        std::vector<float> right_transposed(transposed_count, 0.0F);
        parallel::Config transpose_config;
        transpose_config.minimum_parallel_work = 64U;
        parallel::parallel_for(0U, columns, [&](std::size_t column) {
            for (std::size_t index = 0U; index < inner; ++index) {
                right_transposed[column * inner + index] = right.values()[index * columns + column];
            }
        }, transpose_config);
        const kernel::MatrixView<const float> left_view(left.values().data(), rows, inner, inner, 1U);
        const kernel::MatrixView<const float> right_view(right_transposed.data(), inner, columns, 1U, inner);
        const kernel::MatrixView<float> output_view(output_data.data(), rows, columns, columns, 1U);
    struct AvxDotProduct {
        float operator()(const kernel::MatrixView<const float>& left_view,
                         const kernel::MatrixView<const float>& right_view,
                         std::size_t row,
                         std::size_t column) const noexcept {
            return dot_product(left_view.data() + row * left_view.row_stride(),
                               right_view.data() + column * right_view.column_stride(),
                               left_view.columns());
        }
        };
        kernel::cpu_matmul(left_view, right_view, output_view, AvxDotProduct{});
    }

    Tensor output(Shape{rows, columns}, std::move(output_data), should_track(left, right));
    const auto left_impl = left.internal();
    const auto right_impl = right.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {left_impl, right_impl},
                [left_impl, right_impl, output_impl, rows, inner, columns]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr) {
                        return;
                    }
                    parallel::Config gradient_config;
                    gradient_config.minimum_parallel_work = 64U;
                    if (left_impl->requires_grad) {
                        parallel::parallel_for(0U, rows, [&](std::size_t row) {
                        for (std::size_t index = 0U; index < inner; ++index) {
                            float amount = 0.0F;
                            for (std::size_t column = 0U; column < columns; ++column) {
                                amount += result->grad[row * columns + column] *
                                          right_impl->data[index * columns + column];
                            }
                            left_impl->grad[row * inner + index] += amount;
                        }
                        }, gradient_config);
                    }
                    if (right_impl->requires_grad) {
                        parallel::parallel_for(0U, inner * columns, [&](std::size_t flat_index) {
                            const std::size_t index = flat_index / columns;
                            const std::size_t column = flat_index % columns;
                            float amount = 0.0F;
                            for (std::size_t row = 0U; row < rows; ++row) {
                                amount += left_impl->data[row * inner + index] *
                                          result->grad[row * columns + column];
                            }
                            right_impl->grad[index * columns + column] += amount;
                        }, gradient_config);
                    }
                });
    return output;
}

Tensor relu(const Tensor& value) {
    return unary_elementwise(value, "relu",
                             [](float input) { return std::max(0.0F, input); },
                             [](float input) { return input > 0.0F ? 1.0F : 0.0F; },
                             gpu::ElementwiseOperation::relu);
}

Tensor silu(const Tensor& value) {
    return unary_elementwise(value, "silu",
                             [](float input) {
                                 const float sigmoid_value = 1.0F / (1.0F + std::exp(-input));
                                 return input * sigmoid_value;
                             },
                             [](float input) {
                                 const float sigmoid_value = 1.0F / (1.0F + std::exp(-input));
                                 return sigmoid_value * (1.0F + input * (1.0F - sigmoid_value));
                             },
                             gpu::ElementwiseOperation::silu);
}

Tensor gelu(const Tensor& value) {
    constexpr float coefficient = 0.044715F;
    constexpr float root_two_over_pi = 0.7978845608F;
    return unary_elementwise(value, "gelu",
                             [](float input) {
                                 const float inner = root_two_over_pi *
                                                      (input + coefficient * input * input * input);
                                 return 0.5F * input * (1.0F + std::tanh(inner));
                             },
                             [](float input) {
                                 const float inner = root_two_over_pi *
                                                      (input + coefficient * input * input * input);
                                 const float tanh_inner = std::tanh(inner);
                                 const float derivative_inner = root_two_over_pi *
                                                                (1.0F + 3.0F * coefficient * input * input);
                                 return 0.5F * (1.0F + tanh_inner) +
                                        0.5F * input * (1.0F - tanh_inner * tanh_inner) * derivative_inner;
                             },
                             gpu::ElementwiseOperation::gelu);
}

Tensor sigmoid(const Tensor& value) {
    return unary_elementwise(value, "sigmoid",
                             [](float input) { return 1.0F / (1.0F + std::exp(-input)); },
                             [](float input) {
                                 const float result = 1.0F / (1.0F + std::exp(-input));
                                 return result * (1.0F - result);
                             },
                             gpu::ElementwiseOperation::sigmoid);
}

Tensor sum(const Tensor& value) {
    require_defined(value, "sum");
    const std::size_t worker_count = parallel::worker_count_for(value.numel());
    std::vector<float> partials(worker_count, 0.0F);
    parallel::Config reduction_config;
    reduction_config.worker_count = worker_count;
    reduction_config.grain_size = 1U;
    reduction_config.minimum_parallel_work = 1U;
    parallel::parallel_for(0U, worker_count, [&](std::size_t worker) {
        const std::size_t begin = value.numel() * worker / worker_count;
        const std::size_t end = value.numel() * (worker + 1U) / worker_count;
        partials[worker] = std::accumulate(value.values().begin() +
                                               static_cast<std::ptrdiff_t>(begin),
                                           value.values().begin() +
                                               static_cast<std::ptrdiff_t>(end), 0.0F);
    }, reduction_config);
    const float result = std::accumulate(partials.begin(), partials.end(), 0.0F);
    Tensor output = Tensor::scalar(result, should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl]() {
        const auto result_impl = output_impl.lock();
        if (result_impl == nullptr || !value_impl->requires_grad) {
            return;
        }
        parallel::Config parallel_config;
        parallel_config.minimum_parallel_work = 2048U;
        parallel::parallel_for(0U, value_impl->grad.size(), [&](std::size_t index) {
            value_impl->grad[index] += result_impl->grad[0U];
        }, parallel_config);
    });
    return output;
}

Tensor mean(const Tensor& value) {
    require_defined(value, "mean");
    return sum(value) / static_cast<float>(value.numel());
}

Tensor softmax(const Tensor& value) {
    require_defined(value, "softmax");
    if (value.ndim() == 0U) {
        return Tensor::ones(Shape{}, should_track(value));
    }

    const std::size_t width = value.shape().back();
    if (width == 0U) {
        throw std::invalid_argument("softmax cannot operate on an empty final dimension");
    }
    const std::size_t rows = value.numel() / width;
    std::vector<float> output_data(value.numel(), 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 64U;
    parallel::parallel_for(0U, rows, [&](std::size_t row) {
        const auto begin = value.values().begin() + static_cast<std::ptrdiff_t>(row * width);
        const auto end = begin + static_cast<std::ptrdiff_t>(width);
        const float maximum = *std::max_element(begin, end);
        float denominator = 0.0F;
        for (std::size_t column = 0U; column < width; ++column) {
            const float exponent = std::exp(value.values()[row * width + column] - maximum);
            output_data[row * width + column] = exponent;
            denominator += exponent;
        }
        for (std::size_t column = 0U; column < width; ++column) {
            output_data[row * width + column] /= denominator;
        }
    }, parallel_config);

    Tensor output(value.shape(), std::move(output_data), should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, width]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        const std::size_t gradient_rows = result->data.size() / width;
        parallel::Config gradient_config;
        gradient_config.minimum_parallel_work = 64U;
        parallel::parallel_for(0U, gradient_rows, [&](std::size_t row) {
            float weighted_gradient = 0.0F;
            for (std::size_t column = 0U; column < width; ++column) {
                weighted_gradient += result->grad[row * width + column] *
                                     result->data[row * width + column];
            }
            for (std::size_t column = 0U; column < width; ++column) {
                const float probability = result->data[row * width + column];
                value_impl->grad[row * width + column] += probability *
                                                          (result->grad[row * width + column] -
                                                           weighted_gradient);
            }
        }, gradient_config);
    });
    return output;
}

Tensor cross_entropy(const Tensor& logits, const std::vector<std::size_t>& targets) {
    require_defined(logits, "cross_entropy");
    if (logits.ndim() != 2U || logits.shape()[0] != targets.size() || logits.shape()[1] == 0U) {
        throw std::invalid_argument("cross_entropy requires [batch, classes] logits and one target per row");
    }

    const std::size_t batch = logits.shape()[0];
    const std::size_t classes = logits.shape()[1];
    std::vector<float> probabilities(logits.numel(), 0.0F);
    std::vector<float> row_losses(batch, 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, batch, [&](std::size_t row) {
        if (targets[row] >= classes) {
            throw std::out_of_range("cross_entropy target is outside the class dimension");
        }
        const auto begin = logits.values().begin() + static_cast<std::ptrdiff_t>(row * classes);
        const auto end = begin + static_cast<std::ptrdiff_t>(classes);
        const float maximum = *std::max_element(begin, end);
        float denominator = 0.0F;
        for (std::size_t column = 0U; column < classes; ++column) {
            const float exponent = std::exp(logits.values()[row * classes + column] - maximum);
            probabilities[row * classes + column] = exponent;
            denominator += exponent;
        }
        row_losses[row] = std::log(denominator) + maximum -
                          logits.values()[row * classes + targets[row]];
        for (std::size_t column = 0U; column < classes; ++column) {
            probabilities[row * classes + column] /= denominator;
        }
    }, parallel_config);
    const float loss = std::accumulate(row_losses.begin(), row_losses.end(), 0.0F) /
                       static_cast<float>(batch);

    Tensor output = Tensor::scalar(loss, should_track(logits));
    const auto logits_impl = logits.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {logits_impl},
                [logits_impl, output_impl, probabilities = std::move(probabilities), targets, batch, classes]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr || !logits_impl->requires_grad) {
                        return;
                    }
                    const float scale = result->grad[0U] / static_cast<float>(batch);
                    parallel::Config gradient_config;
                    gradient_config.minimum_parallel_work = 32U;
                    parallel::parallel_for(0U, batch, [&](std::size_t row) {
                        for (std::size_t column = 0U; column < classes; ++column) {
                            float gradient = probabilities[row * classes + column];
                            if (column == targets[row]) {
                                gradient -= 1.0F;
                            }
                            logits_impl->grad[row * classes + column] += scale * gradient;
                        }
                    }, gradient_config);
                });
    return output;
}

Tensor embedding(const Tensor& weights, const std::vector<std::size_t>& token_ids) {
    require_defined(weights, "embedding");
    if (weights.ndim() != 2U) {
        throw std::invalid_argument("embedding weights must have shape [vocabulary, features]");
    }
    const std::size_t vocabulary = weights.shape()[0];
    const std::size_t features = weights.shape()[1];
    for (const std::size_t token : token_ids) {
        if (token >= vocabulary) {
            throw std::out_of_range("embedding token is outside the vocabulary");
        }
    }
    std::vector<float> output_data(token_ids.size() * features, 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, token_ids.size(), [&](std::size_t token_index) {
        for (std::size_t feature = 0U; feature < features; ++feature) {
            output_data[token_index * features + feature] =
                weights.values()[token_ids[token_index] * features + feature];
        }
    }, parallel_config);

    Tensor output(Shape{token_ids.size(), features}, std::move(output_data), should_track(weights));
    const auto weights_impl = weights.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {weights_impl},
                [weights_impl, output_impl, token_ids, features]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr || !weights_impl->requires_grad) {
                        return;
                    }
                    for (std::size_t token_index = 0U; token_index < token_ids.size(); ++token_index) {
                        for (std::size_t feature = 0U; feature < features; ++feature) {
                            weights_impl->grad[token_ids[token_index] * features + feature] +=
                                result->grad[token_index * features + feature];
                        }
                    }
                });
    return output;
}

Tensor rms_norm(const Tensor& value, const Tensor& weight, float epsilon) {
    require_defined(value, "rms_norm");
    require_defined(weight, "rms_norm");
    if (value.ndim() != 2U || weight.ndim() != 1U || value.shape()[1] != weight.shape()[0]) {
        throw std::invalid_argument("rms_norm requires [rows, features] input and [features] weight");
    }
    if (!(epsilon > 0.0F)) {
        throw std::invalid_argument("rms_norm epsilon must be positive");
    }

    const std::size_t rows = value.shape()[0];
    const std::size_t features = value.shape()[1];
    std::vector<float> output_data(value.numel(), 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 64U;
    parallel::parallel_for(0U, rows, [&](std::size_t row) {
        float mean_square = 0.0F;
        for (std::size_t feature = 0U; feature < features; ++feature) {
            const float input = value.values()[row * features + feature];
            mean_square += input * input;
        }
        const float inverse_rms = 1.0F / std::sqrt(mean_square / static_cast<float>(features) + epsilon);
        for (std::size_t feature = 0U; feature < features; ++feature) {
            output_data[row * features + feature] = value.values()[row * features + feature] *
                                                     inverse_rms * weight.values()[feature];
        }
    }, parallel_config);

    Tensor output(value.shape(), std::move(output_data), should_track(value, weight));
    const auto value_impl = value.internal();
    const auto weight_impl = weight.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl, weight_impl},
                [value_impl, weight_impl, output_impl, features, epsilon]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr) {
                        return;
                    }
                    const std::size_t gradient_rows = value_impl->data.size() / features;
                    for (std::size_t row = 0U; row < gradient_rows; ++row) {
                        float mean_square = 0.0F;
                        for (std::size_t feature = 0U; feature < features; ++feature) {
                            const float input = value_impl->data[row * features + feature];
                            mean_square += input * input;
                        }
                        const float inverse_rms = 1.0F /
                                                  std::sqrt(mean_square / static_cast<float>(features) + epsilon);
                        float correction = 0.0F;
                        for (std::size_t feature = 0U; feature < features; ++feature) {
                            const float output_gradient = result->grad[row * features + feature];
                            if (weight_impl->requires_grad) {
                                weight_impl->grad[feature] += output_gradient *
                                                               value_impl->data[row * features + feature] *
                                                               inverse_rms;
                            }
                            correction += output_gradient * weight_impl->data[feature] *
                                          value_impl->data[row * features + feature];
                        }
                        const float correction_scale = inverse_rms * inverse_rms * inverse_rms /
                                                      static_cast<float>(features);
                        for (std::size_t feature = 0U; feature < features; ++feature) {
                            if (value_impl->requires_grad) {
                                const float direct = result->grad[row * features + feature] *
                                                      weight_impl->data[feature] * inverse_rms;
                                value_impl->grad[row * features + feature] +=
                                    direct - value_impl->data[row * features + feature] *
                                            correction_scale * correction;
                            }
                        }
                    }
                });
    return output;
}

Tensor apply_rope(const Tensor& value,
                  std::size_t rotary_dimension,
                  float theta,
                  std::size_t position_offset) {
    require_defined(value, "apply_rope");
    if (value.ndim() != 4U || rotary_dimension == 0U || rotary_dimension > value.shape()[3] ||
        rotary_dimension % 2U != 0U || !(theta > 1.0F)) {
        throw std::invalid_argument("apply_rope requires rank-4 input, an even rotary dimension, and theta > 1");
    }

    const std::size_t batch = value.shape()[0];
    const std::size_t heads = value.shape()[1];
    const std::size_t sequence = value.shape()[2];
    const std::size_t features = value.shape()[3];
    std::vector<float> output_data = value.values();
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, batch * heads * sequence, [&](std::size_t flat_index) {
        const std::size_t position = flat_index % sequence;
        const std::size_t head = (flat_index / sequence) % heads;
        const std::size_t batch_index = flat_index / (heads * sequence);
        const std::size_t absolute_position = position_offset + position;
        for (std::size_t dimension = 0U; dimension < rotary_dimension; dimension += 2U) {
            const float exponent = static_cast<float>(dimension) /
                                   static_cast<float>(rotary_dimension);
            const float angle = static_cast<float>(absolute_position) /
                                std::pow(theta, exponent);
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const std::size_t base = ((batch_index * heads + head) * sequence + position) * features;
            const float first = value.values()[base + dimension];
            const float second = value.values()[base + dimension + 1U];
            output_data[base + dimension] = first * cosine - second * sine;
            output_data[base + dimension + 1U] = first * sine + second * cosine;
        }
    }, parallel_config);

    Tensor output(value.shape(), std::move(output_data), should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, rotary_dimension, theta, position_offset]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        const std::size_t gradient_batch = value_impl->shape[0];
        const std::size_t gradient_heads = value_impl->shape[1];
        const std::size_t gradient_sequence = value_impl->shape[2];
        const std::size_t gradient_features = value_impl->shape[3];
        parallel::Config rope_gradient_config;
        rope_gradient_config.minimum_parallel_work = 32U;
        parallel::parallel_for(0U, gradient_batch * gradient_heads * gradient_sequence,
                               [&](std::size_t flat_index) {
            const std::size_t position = flat_index % gradient_sequence;
            const std::size_t head = (flat_index / gradient_sequence) % gradient_heads;
            const std::size_t batch_index = flat_index / (gradient_heads * gradient_sequence);
            const std::size_t absolute_position = position_offset + position;
            const std::size_t base =
                ((batch_index * gradient_heads + head) * gradient_sequence + position) * gradient_features;
            for (std::size_t dimension = 0U; dimension < rotary_dimension; dimension += 2U) {
                const float exponent = static_cast<float>(dimension) /
                                       static_cast<float>(rotary_dimension);
                const float angle = static_cast<float>(absolute_position) /
                                    std::pow(theta, exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float gradient_first = result->grad[base + dimension];
                const float gradient_second = result->grad[base + dimension + 1U];
                value_impl->grad[base + dimension] += gradient_first * cosine +
                                                     gradient_second * sine;
                value_impl->grad[base + dimension + 1U] += -gradient_first * sine +
                                                           gradient_second * cosine;
            }
            for (std::size_t dimension = rotary_dimension; dimension < gradient_features; ++dimension) {
                value_impl->grad[base + dimension] += result->grad[base + dimension];
            }
        }, rope_gradient_config);
    });
    return output;
}

Tensor scaled_dot_product_attention(const Tensor& query,
                                    const Tensor& key,
                                    const Tensor& value,
                                    bool causal,
                                    std::size_t query_position_offset) {
    require_defined(query, "scaled_dot_product_attention");
    require_defined(key, "scaled_dot_product_attention");
    require_defined(value, "scaled_dot_product_attention");
    if (query.ndim() != 4U || key.ndim() != 4U || value.ndim() != 4U ||
        query.shape()[0] != key.shape()[0] || key.shape()[0] != value.shape()[0] ||
        query.shape()[1] == 0U || key.shape()[1] == 0U ||
        query.shape()[1] % key.shape()[1] != 0U || query.shape()[3] != key.shape()[3] ||
        key.shape()[2] != value.shape()[2]) {
        throw std::invalid_argument("attention received incompatible [batch, heads, sequence, features] tensors");
    }

    const std::size_t batch = query.shape()[0];
    const std::size_t query_heads = query.shape()[1];
    const std::size_t key_value_heads = key.shape()[1];
    const std::size_t query_length = query.shape()[2];
    const std::size_t key_length = key.shape()[2];
    const std::size_t head_dimension = query.shape()[3];
    const std::size_t value_dimension = value.shape()[3];
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
    const std::size_t probability_count = batch * query_heads * query_length * key_length;
    std::vector<float> probabilities(probability_count, 0.0F);
    std::vector<float> output_data(batch * query_heads * query_length * value_dimension, 0.0F);

    const auto probability_index = [query_heads, query_length, key_length](std::size_t batch_index,
                                                                             std::size_t head,
                                                                             std::size_t query_index,
                                                                             std::size_t key_index) {
        return ((batch_index * query_heads + head) * query_length + query_index) * key_length + key_index;
    };
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 16U;
    parallel::parallel_for(0U, batch * query_heads * query_length,
                           [&](std::size_t flat_index) {
            const std::size_t query_index = flat_index % query_length;
            const std::size_t head = (flat_index / query_length) % query_heads;
            const std::size_t batch_index = flat_index / (query_heads * query_length);
            const std::size_t queries_per_key_value_head = query_heads / key_value_heads;
            const std::size_t key_value_head = head / queries_per_key_value_head;
                const std::size_t last_key = causal
                                                  ? std::min(key_length, query_position_offset + query_index + 1U)
                                                  : key_length;
                if (last_key == 0U) {
                    return;
                }
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t key_index = 0U; key_index < last_key; ++key_index) {
                    float score = 0.0F;
                    for (std::size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                        const std::size_t query_offset =
                            ((batch_index * query_heads + head) * query_length + query_index) * head_dimension;
                        const std::size_t key_offset =
                            ((batch_index * key_value_heads + key_value_head) * key_length + key_index) *
                            head_dimension;
                        score += query.values()[query_offset + dimension] * key.values()[key_offset + dimension];
                    }
                    const float scaled_score = score * scale;
                    probabilities[probability_index(batch_index, head, query_index, key_index)] = scaled_score;
                    maximum = std::max(maximum, scaled_score);
                }
                float denominator = 0.0F;
                for (std::size_t key_index = 0U; key_index < last_key; ++key_index) {
                    const std::size_t probability_offset =
                        probability_index(batch_index, head, query_index, key_index);
                    probabilities[probability_offset] = std::exp(probabilities[probability_offset] - maximum);
                    denominator += probabilities[probability_offset];
                }
                for (std::size_t key_index = 0U; key_index < last_key; ++key_index) {
                    const std::size_t probability_offset =
                        probability_index(batch_index, head, query_index, key_index);
                    probabilities[probability_offset] /= denominator;
                    const std::size_t value_offset =
                        ((batch_index * key_value_heads + key_value_head) * key_length + key_index) * value_dimension;
                    const std::size_t output_offset =
                        ((batch_index * query_heads + head) * query_length + query_index) * value_dimension;
                    for (std::size_t dimension = 0U; dimension < value_dimension; ++dimension) {
                        output_data[output_offset + dimension] += probabilities[probability_offset] *
                                                                  value.values()[value_offset + dimension];
                    }
                }
                           }, parallel_config);

    Tensor output(Shape{batch, query_heads, query_length, value_dimension},
                  std::move(output_data), should_track(query, key) || should_track(value));
    const auto query_impl = query.internal();
    const auto key_impl = key.internal();
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {query_impl, key_impl, value_impl},
                [query_impl, key_impl, value_impl, output_impl, probabilities = std::move(probabilities),
                 causal, query_position_offset, batch, query_heads, key_value_heads, query_length, key_length,
                 head_dimension, value_dimension, scale]() {
                    const auto result = output_impl.lock();
                    if (result == nullptr) {
                        return;
                    }
                    const auto gradient_probability_index =
                        [query_heads, query_length, key_length](std::size_t batch_index,
                                                                std::size_t head,
                                                                std::size_t query_index,
                                                                std::size_t key_index) {
                        return ((batch_index * query_heads + head) * query_length + query_index) * key_length +
                               key_index;
                    };
                    for (std::size_t batch_index = 0U; batch_index < batch; ++batch_index) {
                        for (std::size_t head = 0U; head < query_heads; ++head) {
                            const std::size_t queries_per_key_value_head = query_heads / key_value_heads;
                            const std::size_t key_value_head = head / queries_per_key_value_head;
                            for (std::size_t query_index = 0U; query_index < query_length; ++query_index) {
                                const std::size_t last_key = causal
                                                                  ? std::min(key_length,
                                                                             query_position_offset + query_index + 1U)
                                                                  : key_length;
                                float weighted_gradient = 0.0F;
                                for (std::size_t key_index = 0U; key_index < last_key; ++key_index) {
                                    const std::size_t probability_offset =
                                        gradient_probability_index(batch_index, head, query_index, key_index);
                                    const std::size_t output_offset =
                                        ((batch_index * query_heads + head) * query_length + query_index) *
                                        value_dimension;
                                    const std::size_t value_offset =
                                        ((batch_index * key_value_heads + key_value_head) * key_length + key_index) *
                                        value_dimension;
                                    float probability_gradient = 0.0F;
                                    for (std::size_t dimension = 0U; dimension < value_dimension; ++dimension) {
                                        const float output_gradient = result->grad[output_offset + dimension];
                                        if (value_impl->requires_grad) {
                                            value_impl->grad[value_offset + dimension] +=
                                                probabilities[probability_offset] * output_gradient;
                                        }
                                        probability_gradient += output_gradient * value_impl->data[value_offset + dimension];
                                    }
                                    weighted_gradient += probability_gradient * probabilities[probability_offset];
                                }
                                for (std::size_t key_index = 0U; key_index < last_key; ++key_index) {
                                    const std::size_t probability_offset =
                                        gradient_probability_index(batch_index, head, query_index, key_index);
                                    const float probability_gradient = [&]() {
                                        float dot_product = 0.0F;
                                        const std::size_t output_offset =
                                            ((batch_index * query_heads + head) * query_length + query_index) *
                                            value_dimension;
                                        const std::size_t value_offset =
                                            ((batch_index * key_value_heads + key_value_head) * key_length + key_index) *
                                            value_dimension;
                                        for (std::size_t dimension = 0U; dimension < value_dimension; ++dimension) {
                                            dot_product += result->grad[output_offset + dimension] *
                                                           value_impl->data[value_offset + dimension];
                                        }
                                        return dot_product;
                                    }();
                                    const float score_gradient = probabilities[probability_offset] *
                                                                 (probability_gradient - weighted_gradient);
                                    const std::size_t query_offset =
                                        ((batch_index * query_heads + head) * query_length + query_index) *
                                        head_dimension;
                                    const std::size_t key_offset =
                                        ((batch_index * key_value_heads + key_value_head) * key_length + key_index) *
                                        head_dimension;
                                    if (query_impl->requires_grad) {
                                        for (std::size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                                            query_impl->grad[query_offset + dimension] +=
                                                score_gradient * key_impl->data[key_offset + dimension] * scale;
                                        }
                                    }
                                    if (key_impl->requires_grad) {
                                        for (std::size_t dimension = 0U; dimension < head_dimension; ++dimension) {
                                            key_impl->grad[key_offset + dimension] +=
                                                score_gradient * query_impl->data[query_offset + dimension] * scale;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
    return output;
}

Tensor concatenate_sequence(const Tensor& prefix, const Tensor& suffix) {
    require_defined(suffix, "concatenate_sequence");
    if (!prefix.defined()) {
        return suffix.detach();
    }
    if (prefix.ndim() != 4U || suffix.ndim() != 4U ||
        prefix.shape()[0] != suffix.shape()[0] || prefix.shape()[1] != suffix.shape()[1] ||
        prefix.shape()[3] != suffix.shape()[3]) {
        throw std::invalid_argument("sequence concatenation requires compatible rank-4 tensors");
    }
    const std::size_t batch = prefix.shape()[0];
    const std::size_t heads = prefix.shape()[1];
    const std::size_t prefix_length = prefix.shape()[2];
    const std::size_t suffix_length = suffix.shape()[2];
    const std::size_t features = prefix.shape()[3];
    std::vector<float> output_data((prefix_length + suffix_length) * batch * heads * features, 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, batch * heads * (prefix_length + suffix_length),
                           [&](std::size_t flat_index) {
                const std::size_t position = flat_index % (prefix_length + suffix_length);
                const std::size_t head = (flat_index / (prefix_length + suffix_length)) % heads;
                const std::size_t batch_index = flat_index /
                                                 (heads * (prefix_length + suffix_length));
                const bool from_prefix = position < prefix_length;
                const std::size_t source_position = from_prefix ? position : position - prefix_length;
                const std::size_t source_length = from_prefix ? prefix_length : suffix_length;
                const std::size_t source_offset =
                    ((batch_index * heads + head) * source_length + source_position) * features;
                const std::size_t output_offset =
                    ((batch_index * heads + head) * (prefix_length + suffix_length) + position) * features;
                for (std::size_t feature = 0U; feature < features; ++feature) {
                    output_data[output_offset + feature] =
                        (from_prefix ? prefix.values() : suffix.values())[source_offset + feature];
                }
                           }, parallel_config);
    return Tensor(Shape{batch, heads, prefix_length + suffix_length, features},
                  std::move(output_data), false);
}

Tensor split_attention_heads(const Tensor& value, std::size_t head_count) {
    require_defined(value, "split_attention_heads");
    if (value.ndim() != 2U || head_count == 0U || value.shape()[1] % head_count != 0U) {
        throw std::invalid_argument(
            "split_attention_heads requires [sequence, heads * features] input");
    }
    const std::size_t sequence = value.shape()[0];
    const std::size_t features = value.shape()[1] / head_count;
    std::vector<float> output_data(value.numel(), 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, sequence, [&](std::size_t position) {
        for (std::size_t head = 0U; head < head_count; ++head) {
            for (std::size_t feature = 0U; feature < features; ++feature) {
                output_data[(head * sequence + position) * features + feature] =
                    value.values()[position * value.shape()[1] + head * features + feature];
            }
        }
    }, parallel_config);
    Tensor output(Shape{1U, head_count, sequence, features}, std::move(output_data),
                  should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, sequence, head_count, features]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        const std::size_t projection_width = head_count * features;
        parallel::Config gradient_config;
        gradient_config.minimum_parallel_work = 32U;
        parallel::parallel_for(0U, sequence, [&](std::size_t position) {
            for (std::size_t head = 0U; head < head_count; ++head) {
                for (std::size_t feature = 0U; feature < features; ++feature) {
                    value_impl->grad[position * projection_width + head * features + feature] +=
                        result->grad[(head * sequence + position) * features + feature];
                }
            }
        }, gradient_config);
    });
    return output;
}

Tensor merge_attention_heads(const Tensor& value) {
    require_defined(value, "merge_attention_heads");
    if (value.ndim() != 4U || value.shape()[0] != 1U) {
        throw std::invalid_argument(
            "merge_attention_heads requires [1, heads, sequence, features] input");
    }
    const std::size_t heads = value.shape()[1];
    const std::size_t sequence = value.shape()[2];
    const std::size_t features = value.shape()[3];
    std::vector<float> output_data(value.numel(), 0.0F);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 32U;
    parallel::parallel_for(0U, sequence, [&](std::size_t position) {
        for (std::size_t head = 0U; head < heads; ++head) {
            for (std::size_t feature = 0U; feature < features; ++feature) {
                output_data[position * heads * features + head * features + feature] =
                    value.values()[(head * sequence + position) * features + feature];
            }
        }
    }, parallel_config);
    Tensor output(Shape{sequence, heads * features}, std::move(output_data), should_track(value));
    const auto value_impl = value.internal();
    const std::weak_ptr<detail::TensorImpl> output_impl = output.internal();
    attach_node(output, {value_impl}, [value_impl, output_impl, heads, sequence, features]() {
        const auto result = output_impl.lock();
        if (result == nullptr || !value_impl->requires_grad) {
            return;
        }
        parallel::Config gradient_config;
        gradient_config.minimum_parallel_work = 32U;
        parallel::parallel_for(0U, sequence, [&](std::size_t position) {
            for (std::size_t head = 0U; head < heads; ++head) {
                for (std::size_t feature = 0U; feature < features; ++feature) {
                    value_impl->grad[(head * sequence + position) * features + feature] +=
                        result->grad[position * heads * features + head * features + feature];
                }
            }
        }, gradient_config);
    });
    return output;
}

std::size_t argmax_last_row(const Tensor& value) {
    require_defined(value, "argmax_last_row");
    if (value.ndim() != 2U || value.shape()[1] == 0U) {
        throw std::invalid_argument("argmax_last_row requires a rank-2 tensor with a non-empty final dimension");
    }
    const std::size_t width = value.shape()[1];
    const std::size_t start = (value.shape()[0] - 1U) * width;
    return static_cast<std::size_t>(std::distance(
        value.values().begin() + static_cast<std::ptrdiff_t>(start),
        std::max_element(value.values().begin() + static_cast<std::ptrdiff_t>(start), value.values().end())));
}

float l2_norm(const std::vector<Tensor*>& parameters) {
    std::vector<double> partials(parameters.size(), 0.0);
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 1U;
    parallel::parallel_for(0U, parameters.size(), [&](std::size_t parameter_index) {
        const Tensor* parameter = parameters[parameter_index];
        if (parameter == nullptr || !parameter->requires_grad()) {
            return;
        }
        for (const float gradient : parameter->grad_values()) {
            partials[parameter_index] += static_cast<double>(gradient) *
                                         static_cast<double>(gradient);
        }
    }, parallel_config);
    const double sum_of_squares = std::accumulate(partials.begin(), partials.end(), 0.0);
    return static_cast<float>(std::sqrt(sum_of_squares));
}

void clip_grad_norm(const std::vector<Tensor*>& parameters, float maximum_norm) {
    if (!(maximum_norm > 0.0F)) {
        throw std::invalid_argument("maximum gradient norm must be positive");
    }
    const float norm = l2_norm(parameters);
    if (norm <= maximum_norm || norm == 0.0F) {
        return;
    }
    const float scale = maximum_norm / norm;
    parallel::Config parallel_config;
    parallel_config.minimum_parallel_work = 1U;
    parallel::parallel_for(0U, parameters.size(), [&](std::size_t parameter_index) {
        Tensor* parameter = parameters[parameter_index];
        if (parameter == nullptr || !parameter->requires_grad()) {
            return;
        }
        for (float& gradient : parameter->internal()->grad) {
            gradient *= scale;
        }
    }, parallel_config);
}

}  // namespace agentari::nn
