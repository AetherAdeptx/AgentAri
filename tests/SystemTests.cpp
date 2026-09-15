#include "agentari/BitPacking.hpp"
#include "agentari/Kernel.hpp"
#include "agentari/HierarchicalPrediction.hpp"
#include "agentari/Neuron.hpp"
#include "agentari/NeuronMix.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/RunLog.hpp"
#include "agentari/System.hpp"

#include <chrono>
#include <cmath>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
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

class TestCustomNeuron final : public agentari::neuron::Neuron {
public:
    explicit TestCustomNeuron(const std::uint64_t seed) : seed_(seed) {}

    void reset_state() noexcept override { seed_ = 0U; }

private:
    std::uint64_t seed_{0U};
};

void bit_packing_round_trip() {
    using namespace agentari::bits;
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
    const agentari::system::CpuCapabilities capabilities =
        agentari::system::detect_cpu_capabilities();
    require(capabilities.topology.logical_processors >= 1U,
            "CPU detector returned no logical processors");
    require(capabilities.topology.physical_processors >= 1U,
            "CPU detector returned no physical processors");
    require(capabilities.topology.usable_processors >= 1U &&
                capabilities.topology.usable_processors <= capabilities.topology.logical_processors,
            "CPU affinity detector returned an invalid usable count");
    require(std::string(agentari::system::simd_name(capabilities.simd)).size() > 0U,
            "SIMD detector returned an empty name");
    const std::size_t automatic_workers = agentari::parallel::automatic_worker_count();
    require(automatic_workers >= 1U &&
                automatic_workers <= capabilities.topology.usable_processors,
            "automatic worker policy returned an invalid count");
    agentari::parallel::set_max_cpu_usage_percent(50U);
    require(agentari::parallel::available_worker_count() >= 1U &&
                agentari::parallel::available_worker_count() <=
                    capabilities.topology.usable_processors,
            "CPU usage budget returned an invalid worker cap");
    agentari::parallel::set_max_cpu_usage_percent(100U);
    agentari::parallel::set_default_worker_count(2U);
    require(agentari::parallel::default_worker_count() <= 2U,
            "configurable worker policy was not applied");
    agentari::parallel::set_default_worker_count(0U);
    const agentari::system::GpuCapabilities gpu =
        agentari::system::detect_gpu_capabilities();
    require((!gpu.vulkan_candidate || gpu.device_count > 0U),
            "GPU detector returned an invalid candidate report");

    std::vector<std::size_t> visits(10000U, 0U);
    agentari::parallel::Config config;
    config.worker_count = 0U;
    config.grain_size = 17U;
    config.minimum_parallel_work = 1U;
    // Exercise repeated invocations: the worker threads should remain alive
    // and reusable rather than being recreated for every kernel call.
    for (std::size_t round = 0U; round < 8U; ++round) {
        agentari::parallel::parallel_for(
            0U, visits.size(), [&visits](std::size_t index) { ++visits[index]; }, config);
    }
    for (const std::size_t visit : visits) {
        require(visit == 8U, "persistent parallel scheduler skipped or duplicated work");
    }
}

void configurable_neurons() {
    using Basic = agentari::neuron::BasicNeuron<4U, 2U, 3U>;
    using Modern = agentari::neuron::ModernGatedNeuron<4U, 4U, 4U>;
    using Compact = agentari::neuron::CompactModernNeuron<4U, 4U, 4U>;
    using Binary = agentari::neuron::BinaryInferenceNeuron<4U, 2U, 3U>;
    using Sparse = agentari::neuron::SparseModernNeuron<4U, 4U, 2U>;
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

void configurable_neuron_mix() {
    using agentari::neuron::NeuronMixConfig;
    using agentari::neuron::NeuronType;

    NeuronMixConfig partial;
    partial.add(NeuronType::basic, 5.0F);
    partial.add(NeuronType::modern_gated, 4.0F);
    const auto partial_allocation = partial.allocate(1000U);
    require(partial_allocation.entry_count == 2U &&
                partial_allocation.entries[0U].neuron_count == 50U &&
                partial_allocation.entries[1U].neuron_count == 40U &&
                partial_allocation.assigned_neurons == 90U &&
                partial_allocation.unassigned_neurons == 910U,
            "partial neuron mix allocation did not preserve its remainder");

    const auto complete_allocation =
        agentari::neuron::default_neuron_mix().allocate(1000U);
    require(complete_allocation.entry_count == 5U &&
                complete_allocation.assigned_neurons == 1000U &&
                complete_allocation.unassigned_neurons == 0U,
            "complete neuron mix allocation did not consume the network budget");
    require(agentari::neuron::neuron_type_name(NeuronType::sparse_modern) ==
                "sparse_modern",
            "neuron mix type name was not stable");

    agentari::text::hierarchical::PredictionNetworkConfig network_config;
    network_config.total_neuron_count = 1000U;
    network_config.neurons_per_layer = 1U;
    network_config.additional_layer_count = 0U;
    network_config.min_neurons_per_layer = 1U;
    network_config.neuron_mix = agentari::neuron::default_neuron_mix();
    agentari::text::hierarchical::HierarchicalPredictionNetwork network(network_config);
    require(network.neuron_mix_allocation().assigned_neurons == 1000U,
            "prediction network did not load its neuron mix");

    NeuronMixConfig custom;
    custom.add_custom_fixed<TestCustomNeuron>("test_custom", 2U, 0U);
    custom.add_percentage(NeuronType::basic, 100.0F);
    agentari::text::hierarchical::PredictionNetworkConfig custom_config;
    custom_config.total_neuron_count = 16U;
    custom_config.neurons_per_layer = 1U;
    custom_config.additional_layer_count = 0U;
    custom_config.min_neurons_per_layer = 1U;
    custom_config.neuron_mix = custom;
    agentari::text::hierarchical::HierarchicalPredictionNetwork custom_network(custom_config);
    require(custom_network.instantiated_neuron_count() == 16U &&
                custom_network.neurons()[0U] != nullptr,
            "custom inherited neuron factory did not construct the neuron bank");
}

void scalable_cpu_matrix() {
    using namespace agentari::nn::kernel;
    F32CpuMatrix left(64U, 64U, 1.0F);
    F32CpuMatrix right(64U, 64U, 2.0F);
    const F32CpuMatrix result = left.matmul(right);
    require(std::fabs(result(0U, 0U) - 128.0F) < 1.0e-5F,
            "parallel CPU matrix result mismatch");
    require(std::fabs(result(63U, 63U) - 128.0F) < 1.0e-5F,
            "parallel CPU matrix tail result mismatch");
}

void transient_run_log() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("agentari-run-log-test-" + std::to_string(stamp) + ".log");
    {
        std::ofstream stale(path);
        stale << "stale entry\n";
    }
    {
        agentari::diagnostics::RunLog log(path);
        require(log.healthy(), "transient run log did not open");
        log.info("fresh entry");
    }
    std::ifstream first(path);
    const std::string first_contents((std::istreambuf_iterator<char>(first)),
                                     std::istreambuf_iterator<char>());
    require(first_contents.find("stale entry") == std::string::npos,
            "run log retained the previous run");
    require(first_contents.find("fresh entry") != std::string::npos,
            "run log did not record the current run");

    {
        agentari::diagnostics::RunLog log(path);
        require(log.healthy(), "transient run log could not be reopened");
        log.info("second entry");
    }
    std::ifstream second(path);
    const std::string second_contents((std::istreambuf_iterator<char>(second)),
                                      std::istreambuf_iterator<char>());
    require(second_contents.find("fresh entry") == std::string::npos,
            "run log was not cleared on the next startup");
    require(second_contents.find("second entry") != std::string::npos,
            "reopened run log did not record the new run");
    std::error_code filesystem_error;
    std::filesystem::remove(path, filesystem_error);
    require(!filesystem_error, "run log test cleanup failed");
}

}  // namespace

int main() {
    try {
        bit_packing_round_trip();
        topology_and_parallel_execution();
        scalable_cpu_matrix();
        transient_run_log();
        configurable_neurons();
        configurable_neuron_mix();
        std::cout << "agentari system tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "agentari system tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
