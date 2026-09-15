#pragma once

#include "agentari/Neuron.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace agentari::neuron {

// Runtime descriptors let a model combine compile-time neuron presets without
// putting different C++ template instantiations into one unsafe raw array.
enum class NeuronType : std::uint8_t {
    basic,
    modern_gated,
    compact_modern,
    binary_inference,
    sparse_modern,
};

inline constexpr std::string_view neuron_type_name(const NeuronType type) noexcept {
    switch (type) {
    case NeuronType::basic:
        return "basic";
    case NeuronType::modern_gated:
        return "modern_gated";
    case NeuronType::compact_modern:
        return "compact_modern";
    case NeuronType::binary_inference:
        return "binary_inference";
    case NeuronType::sparse_modern:
        return "sparse_modern";
    }
    return "unknown";
}

struct NeuronMixEntry {
    NeuronType type{NeuronType::basic};
    float fill_percent{0.0F};
};

struct NeuronMixAllocationEntry {
    NeuronType type{NeuronType::basic};
    float fill_percent{0.0F};
    std::size_t neuron_count{0U};
};

inline constexpr std::size_t max_neuron_mix_entries = 16U;

struct NeuronMixAllocation {
    std::array<NeuronMixAllocationEntry, max_neuron_mix_entries> entries{};
    std::size_t entry_count{0U};
    std::size_t assigned_neurons{0U};
    std::size_t unassigned_neurons{0U};
};

struct NeuronMixConfig {
    std::array<NeuronMixEntry, max_neuron_mix_entries> entries{};
    std::size_t entry_count{0U};

    void add(const NeuronType type, const float fill_percent) {
        if (entry_count >= entries.size()) {
            throw std::length_error("neuron mix has reached its entry capacity");
        }
        if (!(fill_percent > 0.0F) || !std::isfinite(fill_percent)) {
            throw std::invalid_argument("neuron mix percentages must be finite and positive");
        }
        for (std::size_t index = 0U; index < entry_count; ++index) {
            if (entries[index].type == type) {
                throw std::invalid_argument("neuron mix cannot contain duplicate neuron types");
            }
        }
        entries[entry_count++] = NeuronMixEntry{.type = type, .fill_percent = fill_percent};
    }

    [[nodiscard]] float total_fill_percent() const noexcept {
        float total = 0.0F;
        const std::size_t count = std::min(entry_count, entries.size());
        for (std::size_t index = 0U; index < count; ++index) {
            total += entries[index].fill_percent;
        }
        return total;
    }

    // Partial mixes are allowed so experiments can reserve an explicit
    // unassigned remainder. The default model mix is complete at 100 percent.
    void validate(const bool require_complete = false) const {
        if (entry_count == 0U || entry_count > entries.size()) {
            throw std::invalid_argument("neuron mix must contain at least one entry");
        }

        float total = 0.0F;
        for (std::size_t index = 0U; index < entry_count; ++index) {
            const NeuronMixEntry& entry = entries[index];
            if (!(entry.fill_percent > 0.0F) || !std::isfinite(entry.fill_percent)) {
                throw std::invalid_argument(
                    "neuron mix percentages must be finite and positive");
            }
            for (std::size_t previous = 0U; previous < index; ++previous) {
                if (entries[previous].type == entry.type) {
                    throw std::invalid_argument(
                        "neuron mix cannot contain duplicate neuron types");
                }
            }
            total += entry.fill_percent;
        }
        if (total > 100.0F + 1.0e-3F) {
            throw std::invalid_argument("neuron mix percentages cannot exceed 100 percent");
        }
        if (require_complete && std::fabs(total - 100.0F) > 1.0e-3F) {
            throw std::invalid_argument("complete neuron mix percentages must total 100 percent");
        }
    }

    [[nodiscard]] NeuronMixAllocation allocate(const std::size_t total_neurons) const {
        if (total_neurons == 0U) {
            throw std::invalid_argument("cannot allocate a neuron mix into an empty network");
        }
        validate(false);

        NeuronMixAllocation allocation;
        allocation.entry_count = entry_count;
        std::array<long double, max_neuron_mix_entries> remainders{};
        std::size_t assigned = 0U;
        for (std::size_t index = 0U; index < entry_count; ++index) {
            const NeuronMixEntry& entry = entries[index];
            const long double exact =
                static_cast<long double>(total_neurons) *
                static_cast<long double>(entry.fill_percent) / 100.0L;
            const std::size_t floor_count = static_cast<std::size_t>(std::floor(exact));
            allocation.entries[index] = NeuronMixAllocationEntry{
                .type = entry.type,
                .fill_percent = entry.fill_percent,
                .neuron_count = floor_count,
            };
            remainders[index] = exact - static_cast<long double>(floor_count);
            assigned += floor_count;
        }

        // For a complete mix, largest-remainder rounding makes the typed
        // counts consume exactly the network budget. Partial mixes keep their
        // remainder visible instead of silently assigning it to a different type.
        if (total_fill_percent() >= 100.0F - 1.0e-3F) {
            while (assigned < total_neurons) {
                std::size_t best = 0U;
                for (std::size_t index = 1U; index < entry_count; ++index) {
                    if (remainders[index] > remainders[best]) {
                        best = index;
                    }
                }
                ++allocation.entries[best].neuron_count;
                remainders[best] = -1.0L;
                ++assigned;
            }
        }
        allocation.assigned_neurons = assigned;
        allocation.unassigned_neurons = total_neurons - assigned;
        return allocation;
    }
};

inline NeuronMixConfig default_neuron_mix() {
    NeuronMixConfig mix;
    mix.add(NeuronType::basic, 10.0F);
    mix.add(NeuronType::modern_gated, 35.0F);
    mix.add(NeuronType::compact_modern, 20.0F);
    mix.add(NeuronType::binary_inference, 10.0F);
    mix.add(NeuronType::sparse_modern, 25.0F);
    return mix;
}

}  // namespace agentari::neuron
