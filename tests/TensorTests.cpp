#include "agentari/NeuralNetwork.hpp"
#include "agentari/Kernel.hpp"
#include "agentari/VulkanBackend.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void tensor_math_and_autograd() {
    using namespace agentari::nn;
    Tensor left({2U, 2U}, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, true);
    Tensor right({2U, 2U}, std::vector<float>{2.0F, 0.0F, 1.0F, 2.0F}, true);
    const Tensor product = matmul(left, right);
    require_near(product.at({0U, 0U}), 4.0F, 1.0e-6F, "matmul value mismatch");
    require_near(product.at({1U, 1U}), 8.0F, 1.0e-6F, "matmul value mismatch");

    const Tensor loss = sum(product);
    loss.backward();
    require_near(left.grad_values()[0U], 2.0F, 1.0e-6F, "matmul left gradient mismatch");
    require_near(left.grad_values()[1U], 3.0F, 1.0e-6F, "matmul left gradient mismatch");
    require_near(right.grad_values()[0U], 4.0F, 1.0e-6F, "matmul right gradient mismatch");
    require_near(right.grad_values()[3U], 6.0F, 1.0e-6F, "matmul right gradient mismatch");

    Tensor bias({2U}, std::vector<float>{1.0F, 2.0F}, true);
    const Tensor biased = product + bias;
    require_near(biased.at({1U, 0U}), 11.0F, 1.0e-6F, "final-dimension broadcast mismatch");
    require_near(biased.at({1U, 1U}), 10.0F, 1.0e-6F, "final-dimension broadcast mismatch");
}

void policy_matrix_components() {
    using namespace agentari::nn::kernel;
    F32CpuMatrix left(2U, 3U);
    left(0U, 0U) = 1.0F;
    left(0U, 1U) = 2.0F;
    left(0U, 2U) = 3.0F;
    left(1U, 0U) = 4.0F;
    left(1U, 1U) = 5.0F;
    left(1U, 2U) = 6.0F;
    Matrix<float, CpuBackend, ColumnMajor> right(3U, 2U);
    right(0U, 0U) = 7.0F;
    right(0U, 1U) = 8.0F;
    right(1U, 0U) = 9.0F;
    right(1U, 1U) = 10.0F;
    right(2U, 0U) = 11.0F;
    right(2U, 1U) = 12.0F;
    const F32CpuMatrix result = left.matmul(right);
    require_near(result(0U, 0U), 58.0F, 1.0e-6F, "policy matrix product mismatch");
    require_near(result(1U, 1U), 154.0F, 1.0e-6F, "policy matrix product mismatch");
}

void normalization_and_attention() {
    using namespace agentari::nn;
    Tensor input({1U, 2U, 3U, 4U},
                 std::vector<float>{1.0F, 0.0F, 0.0F, 0.0F,
                                    0.0F, 1.0F, 0.0F, 0.0F,
                                    0.0F, 0.0F, 1.0F, 0.0F,
                                    0.0F, 0.0F, 0.0F, 1.0F,
                                    1.0F, 1.0F, 0.0F, 0.0F,
                                    0.0F, 1.0F, 1.0F, 0.0F},
                 true);
    Tensor query = apply_rope(input, 4U, 10000.0F);
    Tensor key = apply_rope(input, 4U, 10000.0F);
    Tensor value = input;
    const Tensor attended = scaled_dot_product_attention(query, key, value, true);
    require(attended.shape() == Shape{1U, 2U, 3U, 4U}, "attention shape mismatch");
    const Tensor normalized = rms_norm(input.reshape({6U, 4U}), Tensor::ones({4U}, true), 1.0e-5F);
    require(normalized.shape() == Shape{6U, 4U}, "RMSNorm shape mismatch");
    const Tensor loss = mean(attended) + mean(normalized);
    loss.backward();
    require(input.has_grad(), "attention and normalization did not produce gradients");
}

void transformer_forward_backward_and_generation() {
    using namespace agentari::nn;
    TransformerConfig config;
    config.vocabulary_size = 16U;
    config.maximum_sequence_length = 16U;
    config.model_dimension = 16U;
    config.layer_count = 1U;
    config.attention_head_count = 4U;
    config.key_value_head_count = 2U;
    config.feed_forward_dimension = 32U;
    config.seed = 123U;

    TransformerModel model(config);
    const std::vector<std::size_t> input{1U, 2U, 3U, 4U};
    const std::vector<std::size_t> target{2U, 3U, 4U, 5U};
    const Tensor logits = model.forward(input);
    require(logits.shape() == Shape{4U, 16U}, "Transformer logits shape mismatch");
    {
        NoGradGuard no_grad;
        std::vector<KeyValueCache> cache(config.layer_count);
        for (std::size_t position = 0U; position < input.size(); ++position) {
            const Tensor step_logits = model.step(input[position], cache, position);
            for (std::size_t token = 0U; token < config.vocabulary_size; ++token) {
                require_near(step_logits.values()[token],
                             logits.values()[position * config.vocabulary_size + token],
                             1.0e-5F,
                             "cached decoding differs from full-sequence attention");
            }
        }
    }
    const Tensor loss = model.loss(input, target);
    require(std::isfinite(loss.values()[0U]), "Transformer loss is not finite");
    loss.backward();

    auto parameters = model.parameters();
    require(!parameters.empty(), "Transformer did not expose parameters");
    require(l2_norm(parameters) > 0.0F, "Transformer gradients are empty");
    clip_grad_norm(parameters, 1.0F);
    AdamW optimizer(parameters, 1.0e-3F);
    optimizer.step();
    optimizer.zero_grad();
    require(optimizer.update_count() == 1U, "AdamW update count mismatch");

    const auto generated = model.generate(input, GenerationConfig{
        .maximum_new_tokens = 3U,
        .temperature = 0.0F,
        .top_k = 0U,
        .seed = 9U,
    });
    require(generated.size() == 7U, "generation length mismatch");
    require(model.parameter_count() == 2736U, "parameter count mismatch");
}

void optional_vulkan_tensor_backend() {
    using namespace agentari::nn;
    agentari::gpu::VulkanComputeBackend gpu_backend;
    set_tensor_backend(TensorBackend::vulkan, &gpu_backend);
    Tensor left({16U, 16U}, 1.0F);
    Tensor right({16U, 16U}, 2.0F);
    const Tensor product = matmul(left, right);
    require_near(product.at({0U, 0U}), 32.0F, 1.0e-4F,
                 "optional Vulkan tensor matmul mismatch");
    require_near(product.at({15U, 15U}), 32.0F, 1.0e-4F,
                 "optional Vulkan tensor tail mismatch");
    if (gpu_backend.available()) {
        require(!gpu_backend.device_name().empty(),
                "available Vulkan backend did not report a device");
        const float gpu_left[4] = {1.0F, -2.0F, 3.0F, 4.0F};
        const float gpu_right[4] = {2.0F, 5.0F, -1.0F, 2.0F};
        float gpu_output[4]{};
        require(gpu_backend.elementwise(agentari::gpu::ElementwiseOperation::add,
                                        gpu_left, gpu_right, 4U, gpu_output),
                "Vulkan elementwise dispatch failed");
        require_near(gpu_output[0U], 3.0F, 1.0e-4F,
                     "Vulkan elementwise result mismatch");
        require_near(gpu_output[2U], 2.0F, 1.0e-4F,
                     "Vulkan elementwise tail mismatch");
    }
    set_tensor_backend(TensorBackend::cpu, nullptr);
    require(tensor_backend() == TensorBackend::cpu,
            "tensor backend did not restore CPU selection");
}

}  // namespace

int main() {
    try {
        tensor_math_and_autograd();
        policy_matrix_components();
        normalization_and_attention();
        transformer_forward_backward_and_generation();
        optional_vulkan_tensor_backend();
        std::cout << "agentari tensor tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "agentari tensor tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
