#include "firstagent/BitPacking.hpp"
#include "firstagent/Kernel.hpp"
#include "firstagent/Neuron.hpp"
#include "firstagent/Parallel.hpp"
#include "firstagent/System.hpp"

#include <cmath>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void bit_packing_round_trip() {
    using namespace firstagent::bits;
    require(align_up(0U, 64U) == 0U, "zero alignment mismatch");
    require(align_up(65U, 64U) == 128U, "cache-line alignment mismatch");
    require(bits_required(0U) == 1U && bits_required(255U) == 8U,
            "bit-width calculation mismatch");

    BitWriter writer;
    writer.write(1U, 1U);
    writer.write(3U, 2U);
    writer.write(17U, 5U);
    writer.write(std::numeric_limits<std::uint64_t>::max(), 64U);
    BitReader reader(std::span<const std::byte>(writer.bytes()), writer.bit_count());
    require(reader.read(1U) == 1U, "one-bit round trip mismatch");
    require(reader.read(2U) == 3U, "two-bit round trip mismatch");
    require(reader.read(5U) == 17U, "five-bit round trip mismatch");
    require(reader.read(64U) == std::numeric_limits<std::uint64_t>::max(),
            "64-bit round trip mismatch");
    require(reader.remaining_bits() == 0U, "packed reader retained bits");

    PackedVector states(2U, 16U);
    for (std::size_t index = 0U; index < states.size(); ++index) {
        states.set(index, index % 4U);
    }
    require(states.bytes().size() == 4U, "two-bit state vector was not packed");
    for (std::size_t index = 0U; index < states.size(); ++index) {
        require(states.get(index) == index % 4U, "packed vector value mismatch");
    }

    AlignedPackedArray<3U> cache_states(257U);
    for (std::size_t index = 0U; index < cache_states.size(); ++index) {
        cache_states.set(index, index % 8U);
    }
    require(cache_states.block_count() >= 1U &&
                cache_states.bytes_reserved() % cache_states.alignment_bytes() == 0U,
            "aligned packed array did not reserve cache-line blocks");
    require(cache_states.get(0U) == 0U && cache_states.get(256U) == 0U,
            "aligned packed array boundary round trip mismatch");
    require(cache_states.get(127U) == 7U, "aligned packed array cross-block mismatch");
}

void topology_and_parallel_execution() {
    const firstagent::system::CpuCapabilities capabilities =
        firstagent::system::detect_cpu_capabilities();
    require(capabilities.topology.logical_processors >= 1U,
            "CPU detector returned no logical processors");
    require(capabilities.topology.physical_processors >= 1U,
            "CPU detector returned no physical processors");
    require(capabilities.topology.usable_processors >= 1U &&
                capabilities.topology.usable_processors <= capabilities.topology.logical_processors,
            "CPU affinity detector returned an invalid usable count");
    require(std::string(firstagent::system::simd_name(capabilities.simd)).size() > 0U,
            "SIMD detector returned an empty name");
    const std::size_t automatic_workers = firstagent::parallel::automatic_worker_count();
    require(automatic_workers >= 1U &&
                automatic_workers <= capabilities.topology.usable_processors,
            "automatic worker policy returned an invalid count");
    firstagent::parallel::set_default_worker_count(2U);
    require(firstagent::parallel::default_worker_count() <= 2U,
            "configurable worker policy was not applied");
    firstagent::parallel::set_default_worker_count(0U);
    const firstagent::system::GpuCapabilities gpu =
        firstagent::system::detect_gpu_capabilities();
    require((!gpu.vulkan_candidate || gpu.device_count > 0U),
            "GPU detector returned an invalid candidate report");

    std::vector<std::size_t> visits(10000U, 0U);
    firstagent::parallel::Config config;
    config.worker_count = 0U;
    config.grain_size = 17U;
    config.minimum_parallel_work = 1U;
    firstagent::parallel::parallel_for(0U, visits.size(),
                                       [&visits](std::size_t index) { ++visits[index]; }, config);
    for (const std::size_t visit : visits) {
        require(visit == 1U, "parallel scheduler skipped or duplicated work");
    }
}

void configurable_neurons() {
    using Basic = firstagent::neuron::BasicNeuron<4U, 2U, 3U>;
    using Modern = firstagent::neuron::ModernGatedNeuron<4U, 4U, 4U>;
    using Compact = firstagent::neuron::CompactModernNeuron<4U, 4U, 4U>;
    using Binary = firstagent::neuron::BinaryInferenceNeuron<4U, 2U, 3U>;
    using Sparse = firstagent::neuron::SparseModernNeuron<4U, 4U, 2U>;
    require(Basic::input_width == 4U && Basic::state_width == 2U &&
                Basic::output_width == 3U,
            "basic neuron template dimensions were not preserved");
    require(Basic::bytes_per_instance() != Modern::bytes_per_instance(),
            "neuron feature templates did not change storage size");
    require(Compact::bytes_per_instance() < Modern::bytes_per_instance(),
            "compact neuron storage did not reduce memory footprint");

    const std::array<float, 4U> input{1.0F, -0.5F, 0.25F, 0.75F};
    const std::array<float, 4U> feedback{0.1F, 0.0F, -0.1F, 0.2F};
    std::array<float, 4U> target{0.5F, 0.0F, -0.5F, 0.25F};
    Modern modern(42U);
    const auto first = modern.forward(std::span<const float>(input),
                                      std::span<const float>(feedback));
    const auto prediction = modern.learn(std::span<const float>(input),
                                         std::span<const float>(target), 0.01F,
                                         std::span<const float>(feedback));
    require(std::isfinite(first[0U]) && std::isfinite(prediction[0U]),
            "modern neuron produced a non-finite activation");
    Binary binary(7U);
    const auto binary_output = binary.forward(std::span<const float>(input));
    require(std::isfinite(binary_output[0U]),
            "binary inference neuron produced a non-finite activation");
    Sparse sparse(99U);
    const auto sparse_output = sparse.forward(std::span<const float>(input),
                                              std::span<const float>(feedback));
    float routing_sum = 0.0F;
    for (const float weight : sparse.routing_weights()) {
        routing_sum += weight;
    }
    require(std::isfinite(sparse_output[0U]) && std::fabs(routing_sum - 1.0F) < 1.0e-5F,
            "sparse mixture neuron routing was invalid");
}

void scalable_cpu_matrix() {
    using namespace firstagent::nn::kernel;
    F32CpuMatrix left(64U, 64U, 1.0F);
    F32CpuMatrix right(64U, 64U, 2.0F);
    const F32CpuMatrix result = left.matmul(right);
    require(std::fabs(result(0U, 0U) - 128.0F) < 1.0e-5F,
            "parallel CPU matrix result mismatch");
    require(std::fabs(result(63U, 63U) - 128.0F) < 1.0e-5F,
            "parallel CPU matrix tail result mismatch");
}

}  // namespace

int main() {
    try {
        bit_packing_round_trip();
        topology_and_parallel_execution();
        scalable_cpu_matrix();
        configurable_neurons();
        std::cout << "firstagent system tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "firstagent system tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
