#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace firstagent::gpu {

enum class ElementwiseOperation : std::uint8_t {
    add,
    subtract,
    multiply,
    relu,
    silu,
    gelu,
    sigmoid,
};

class VulkanComputeBackend {
public:
    explicit VulkanComputeBackend(std::string shader_path = {});
    ~VulkanComputeBackend();

    VulkanComputeBackend(const VulkanComputeBackend&) = delete;
    VulkanComputeBackend& operator=(const VulkanComputeBackend&) = delete;

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;
    [[nodiscard]] const std::string& error() const noexcept;

    // Computes C = A * B for row-major matrices. This is the first GPU
    // primitive; the full autograd tensor graph will be routed here next.
    [[nodiscard]] bool matmul(const float* left,
                              std::size_t rows,
                              std::size_t inner,
                              const float* right,
                              std::size_t columns,
                              float* output) const;

    [[nodiscard]] bool elementwise(ElementwiseOperation operation,
                                   const float* left,
                                   const float* right,
                                   std::size_t count,
                                   float* output) const;

private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace firstagent::gpu
