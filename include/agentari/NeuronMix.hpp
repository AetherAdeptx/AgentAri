#pragma once

#include "agentari/Neuron.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace agentari::neuron {

// Runtime descriptors let a model combine compile-time neuron presets without
// putting different C++ template instantiations into one unsafe raw array.
enum class NeuronType : std::uint8_t {
    basic,
    modern_gated,
    compact_modern,
    binary_inference,
    sparse_modern,
    custom,
};

inline constexpr std::uint32_t neuron_any_layer =
    std::numeric_limits<std::uint32_t>::max();

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
    case NeuronType::custom:
        return "custom";
    }
    return "unknown";
}

enum class NeuronAllocationMode : std::uint8_t {
    percentage,
    fixed_count,
};

using NeuronFactory = std::function<std::unique_ptr<Neuron>(std::uint64_t)>;

struct NeuronMixEntry {
    NeuronType type{NeuronType::basic};
    std::string type_name{};
    NeuronAllocationMode mode{NeuronAllocationMode::percentage};
    std::size_t fixed_count{0U};
    float fill_percent{0.0F};
    // A fixed entry may target one layer. Percent entries may also target a
    // layer; their percentage is still calculated from the global remainder.
    std::uint32_t layer_id{neuron_any_layer};
    // Custom classes provide this factory. Built-in entries receive one
    // automatically; an empty factory falls back to a base Neuron instance.
    NeuronFactory factory{};
};

struct NeuronMixAllocationEntry {
    // Entries are numbered in insertion order, starting at one. Spatial
    // records use this stable number instead of relying on enum values.
    std::uint32_t number{0U};
    NeuronType type{NeuronType::basic};
    std::string type_name{};
    NeuronAllocationMode mode{NeuronAllocationMode::percentage};
    std::size_t fixed_count{0U};
    float fill_percent{0.0F};
    std::uint32_t layer_id{neuron_any_layer};
    std::size_t neuron_count{0U};
};

struct NeuronPlacement {
    std::uint32_t number{0U};
    NeuronType type{NeuronType::basic};
    std::uint32_t layer_id{0U};
};

inline constexpr std::size_t max_neuron_mix_entries = 16U;

struct NeuronMixAllocation {
    std::array<NeuronMixAllocationEntry, max_neuron_mix_entries> entries{};
    std::size_t entry_count{0U};
    std::size_t assigned_neurons{0U};
    std::size_t unassigned_neurons{0U};
    // Only populated when allocation is given layer capacities. The vector
    // contains one record per explicitly typed neuron, not unassigned slots.
    std::vector<NeuronPlacement> placements{};
};

inline NeuronFactory builtin_neuron_factory(const NeuronType type) {
    switch (type) {
    case NeuronType::basic:
        return [](const std::uint64_t seed) {
            return std::make_unique<BasicNeuron<1U, 1U, 1U>>(seed);
        };
    case NeuronType::modern_gated:
        return [](const std::uint64_t seed) {
            return std::make_unique<ModernGatedNeuron<1U, 1U, 1U>>(seed);
        };
    case NeuronType::compact_modern:
        return [](const std::uint64_t seed) {
            return std::make_unique<CompactModernNeuron<1U, 1U, 1U>>(seed);
        };
    case NeuronType::binary_inference:
        return [](const std::uint64_t seed) {
            return std::make_unique<BinaryInferenceNeuron<1U, 1U, 1U>>(seed);
        };
    case NeuronType::sparse_modern:
        return [](const std::uint64_t seed) {
            return std::make_unique<SparseModernNeuron<1U, 4U, 2U>>(seed);
        };
    case NeuronType::custom:
        return {};
    }
    return {};
}

struct NeuronMixConfig {
    std::array<NeuronMixEntry, max_neuron_mix_entries> entries{};
    std::size_t entry_count{0U};

    // Backwards-compatible shorthand for a percentage entry.
    void add(const NeuronType type,
             const float fill_percent,
             const std::uint32_t layer_id = neuron_any_layer) {
        add_percentage(type, fill_percent, layer_id);
    }

    void add_percentage(const NeuronType type,
                         const float fill_percent,
                         const std::uint32_t layer_id = neuron_any_layer) {
        append(NeuronMixEntry{
            .type = type,
            .type_name = std::string(neuron_type_name(type)),
            .mode = NeuronAllocationMode::percentage,
            .fixed_count = 0U,
            .fill_percent = fill_percent,
            .layer_id = layer_id,
            .factory = builtin_neuron_factory(type),
        });
    }

    void add_fixed(const NeuronType type,
                   const std::size_t count,
                   const std::uint32_t layer_id = neuron_any_layer) {
        append(NeuronMixEntry{
            .type = type,
            .type_name = std::string(neuron_type_name(type)),
            .mode = NeuronAllocationMode::fixed_count,
            .fixed_count = count,
            .fill_percent = 0.0F,
            .layer_id = layer_id,
            .factory = builtin_neuron_factory(type),
        });
    }

    template <typename CustomNeuron>
    void add_custom_percentage(std::string type_name,
                               const float fill_percent,
                               const std::uint32_t layer_id = neuron_any_layer) {
        static_assert(std::is_base_of_v<Neuron, CustomNeuron>,
                      "custom neurons must inherit from agentari::neuron::Neuron");
        static_assert(std::is_constructible_v<CustomNeuron, std::uint64_t>,
                      "custom neurons need a uint64_t seed constructor");
        append(NeuronMixEntry{
            .type = NeuronType::custom,
            .type_name = std::move(type_name),
            .mode = NeuronAllocationMode::percentage,
            .fixed_count = 0U,
            .fill_percent = fill_percent,
            .layer_id = layer_id,
            .factory = [](const std::uint64_t seed) {
                return std::make_unique<CustomNeuron>(seed);
            },
        });
    }

    template <typename CustomNeuron>
    void add_custom_fixed(std::string type_name,
                          const std::size_t count,
                          const std::uint32_t layer_id = neuron_any_layer) {
        static_assert(std::is_base_of_v<Neuron, CustomNeuron>,
                      "custom neurons must inherit from agentari::neuron::Neuron");
        static_assert(std::is_constructible_v<CustomNeuron, std::uint64_t>,
                      "custom neurons need a uint64_t seed constructor");
        append(NeuronMixEntry{
            .type = NeuronType::custom,
            .type_name = std::move(type_name),
            .mode = NeuronAllocationMode::fixed_count,
            .fixed_count = count,
            .fill_percent = 0.0F,
            .layer_id = layer_id,
            .factory = [](const std::uint64_t seed) {
                return std::make_unique<CustomNeuron>(seed);
            },
        });
    }

    [[nodiscard]] float total_fill_percent() const noexcept {
        float total = 0.0F;
        const std::size_t count = std::min(entry_count, entries.size());
        for (std::size_t index = 0U; index < count; ++index) {
            if (entries[index].mode == NeuronAllocationMode::percentage) {
                total += entries[index].fill_percent;
            }
        }
        return total;
    }

    [[nodiscard]] std::size_t total_fixed_count() const noexcept {
        std::size_t total = 0U;
        const std::size_t count = std::min(entry_count, entries.size());
        for (std::size_t index = 0U; index < count; ++index) {
            if (entries[index].mode == NeuronAllocationMode::fixed_count) {
                total += entries[index].fixed_count;
            }
        }
        return total;
    }

    // Partial percentage mixes are allowed so experiments can reserve an
    // explicit unassigned remainder. A complete percentage mix consumes the
    // entire post-fixed budget using largest-remainder rounding.
    void validate(const bool require_complete = false) const {
        if (entry_count == 0U || entry_count > entries.size()) {
            throw std::invalid_argument("neuron mix must contain at least one entry");
        }

        float percentage_total = 0.0F;
        std::size_t fixed_total = 0U;
        for (std::size_t index = 0U; index < entry_count; ++index) {
            const NeuronMixEntry& entry = entries[index];
            if (neuron_type_name(entry.type) == "unknown") {
                throw std::invalid_argument("neuron mix contains an unknown neuron type");
            }
            if (entry.type == NeuronType::custom && entry.type_name.empty()) {
                throw std::invalid_argument("custom neuron mix entries require a type name");
            }
            for (std::size_t previous = 0U; previous < index; ++previous) {
                const bool same_builtin = entry.type != NeuronType::custom &&
                                          entries[previous].type == entry.type;
                const bool same_custom = entry.type == NeuronType::custom &&
                                         entries[previous].type == NeuronType::custom &&
                                         entries[previous].type_name == entry.type_name;
                if (same_builtin || same_custom) {
                    throw std::invalid_argument(
                        "neuron mix cannot contain duplicate neuron types");
                }
            }
            if (entry.mode == NeuronAllocationMode::fixed_count) {
                if (entry.fixed_count == 0U) {
                    throw std::invalid_argument(
                        "fixed neuron mix entries must contain at least one neuron");
                }
                if (fixed_total > std::numeric_limits<std::size_t>::max() -
                                      entry.fixed_count) {
                    throw std::invalid_argument("fixed neuron counts overflow the size type");
                }
                fixed_total += entry.fixed_count;
            } else {
                if (!(entry.fill_percent > 0.0F) || !std::isfinite(entry.fill_percent)) {
                    throw std::invalid_argument(
                        "neuron mix percentages must be finite and positive");
                }
                percentage_total += entry.fill_percent;
            }
        }
        if (percentage_total > 100.0F + 1.0e-3F) {
            throw std::invalid_argument("neuron mix percentages cannot exceed 100 percent");
        }
        if (require_complete &&
            std::fabs(percentage_total - 100.0F) > 1.0e-3F) {
            throw std::invalid_argument(
                "complete neuron mix percentages must total 100 percent");
        }
    }

    // Allocates only the counts. This is useful for inspecting a mix before
    // layer geometry exists.
    [[nodiscard]] NeuronMixAllocation allocate(const std::size_t total_neurons) const {
        return allocate_impl(total_neurons, std::span<const std::size_t>{});
    }

    // Allocates counts and deterministic layer assignments. Fixed entries
    // with a layer bias are placed first, followed by unbound fixed entries,
    // then percentage entries from the remaining global budget.
    [[nodiscard]] NeuronMixAllocation allocate(
        const std::size_t total_neurons,
        const std::span<const std::size_t> layer_neuron_counts) const {
        return allocate_impl(total_neurons, layer_neuron_counts);
    }

private:
    void append(const NeuronMixEntry entry) {
        if (entry_count >= entries.size()) {
            throw std::length_error("neuron mix has reached its entry capacity");
        }
        for (std::size_t index = 0U; index < entry_count; ++index) {
            const bool same_builtin = entry.type != NeuronType::custom &&
                                      entries[index].type == entry.type;
            const bool same_custom = entry.type == NeuronType::custom &&
                                     entries[index].type == NeuronType::custom &&
                                     entries[index].type_name == entry.type_name;
            if (same_builtin || same_custom) {
                throw std::invalid_argument("neuron mix cannot contain duplicate neuron types");
            }
        }
        entries[entry_count++] = entry;
    }

    [[nodiscard]] static std::vector<std::size_t>
    symmetric_layer_order(const std::size_t layer_count) {
        std::vector<std::size_t> order;
        order.reserve(layer_count);
        if (layer_count == 0U) {
            return order;
        }
        const std::size_t center = (layer_count - 1U) / 2U;
        order.push_back(center);
        for (std::size_t offset = 1U; offset < layer_count; ++offset) {
            if (center + offset < layer_count) {
                order.push_back(center + offset);
            }
            if (center >= offset) {
                order.push_back(center - offset);
            }
        }
        return order;
    }

    [[nodiscard]] NeuronMixAllocation allocate_impl(
        const std::size_t total_neurons,
        const std::span<const std::size_t> layer_neuron_counts) const {
        if (total_neurons == 0U) {
            throw std::invalid_argument("cannot allocate a neuron mix into an empty network");
        }
        validate(false);

        NeuronMixAllocation allocation;
        allocation.entry_count = entry_count;
        for (std::size_t index = 0U; index < entry_count; ++index) {
            allocation.entries[index] = NeuronMixAllocationEntry{
                .number = static_cast<std::uint32_t>(index + 1U),
                .type = entries[index].type,
                .type_name = entries[index].type_name.empty()
                                 ? std::string(neuron_type_name(entries[index].type))
                                 : entries[index].type_name,
                .mode = entries[index].mode,
                .fixed_count = entries[index].fixed_count,
                .fill_percent = entries[index].fill_percent,
                .layer_id = entries[index].layer_id,
            };
        }

        std::vector<std::size_t> remaining_capacity;
        std::vector<std::size_t> layer_order;
        std::size_t layer_cursor = 0U;
        if (!layer_neuron_counts.empty()) {
            std::size_t capacity_total = 0U;
            remaining_capacity.assign(layer_neuron_counts.begin(),
                                      layer_neuron_counts.end());
            for (const std::size_t capacity : remaining_capacity) {
                if (capacity > std::numeric_limits<std::size_t>::max() - capacity_total) {
                    throw std::invalid_argument("layer capacities overflow the size type");
                }
                capacity_total += capacity;
            }
            if (capacity_total != total_neurons) {
                throw std::invalid_argument(
                    "layer capacities must add up to Total_Neuron_Count");
            }
            if (remaining_capacity.size() >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::invalid_argument("layer count exceeds the layer ID range");
            }
            layer_order = symmetric_layer_order(remaining_capacity.size());
            allocation.placements.reserve(total_neurons);
        }

        auto add_count = [&allocation](const std::size_t entry_index,
                                       const std::size_t count) {
            if (count > std::numeric_limits<std::size_t>::max() -
                            allocation.assigned_neurons ||
                count > std::numeric_limits<std::size_t>::max() -
                            allocation.entries[entry_index].neuron_count) {
                throw std::invalid_argument("assigned neuron count overflowed the size type");
            }
            allocation.entries[entry_index].neuron_count += count;
            allocation.assigned_neurons += count;
        };

        auto append_to_layer = [&](const std::size_t entry_index,
                                   const std::size_t count,
                                   const std::uint32_t requested_layer) {
            if (layer_neuron_counts.empty() || count == 0U) {
                return;
            }
            auto append_one = [&](const std::size_t layer) {
                if (remaining_capacity[layer] == 0U) {
                    throw std::invalid_argument(
                        "layer-biased neuron assignment exceeds layer capacity");
                }
                --remaining_capacity[layer];
                allocation.placements.push_back(NeuronPlacement{
                    .number = allocation.entries[entry_index].number,
                    .type = allocation.entries[entry_index].type,
                    .layer_id = static_cast<std::uint32_t>(layer),
                });
            };

            if (requested_layer != neuron_any_layer) {
                if (requested_layer >= remaining_capacity.size()) {
                    throw std::invalid_argument(
                        "layer-biased neuron assignment references an unknown layer");
                }
                for (std::size_t neuron = 0U; neuron < count; ++neuron) {
                    append_one(static_cast<std::size_t>(requested_layer));
                }
                return;
            }

            for (std::size_t neuron = 0U; neuron < count; ++neuron) {
                bool placed = false;
                for (std::size_t probe = 0U; probe < layer_order.size(); ++probe) {
                    const std::size_t order_index =
                        (layer_cursor + probe) % layer_order.size();
                    const std::size_t layer = layer_order[order_index];
                    if (remaining_capacity[layer] != 0U) {
                        append_one(layer);
                        layer_cursor = (order_index + 1U) % layer_order.size();
                        placed = true;
                        break;
                    }
                }
                if (!placed) {
                    throw std::invalid_argument(
                        "fixed neuron assignments exceed total layer capacity");
                }
            }
        };

        auto assign_entry = [&](const std::size_t entry_index, const std::size_t count) {
            add_count(entry_index, count);
            append_to_layer(entry_index, count, entries[entry_index].layer_id);
        };

        // Fixed, layer-biased entries always reserve their cells first.
        for (std::size_t index = 0U; index < entry_count; ++index) {
            if (entries[index].mode == NeuronAllocationMode::fixed_count &&
                entries[index].layer_id != neuron_any_layer) {
                if (entries[index].fixed_count > total_neurons -
                                                       allocation.assigned_neurons) {
                    throw std::invalid_argument(
                        "fixed neuron assignments exceed Total_Neuron_Count");
                }
                assign_entry(index, entries[index].fixed_count);
            }
        }
        // Then place unbound fixed entries using a symmetric center-out order.
        for (std::size_t index = 0U; index < entry_count; ++index) {
            if (entries[index].mode == NeuronAllocationMode::fixed_count &&
                entries[index].layer_id == neuron_any_layer) {
                if (entries[index].fixed_count > total_neurons -
                                                       allocation.assigned_neurons) {
                    throw std::invalid_argument(
                        "fixed neuron assignments exceed Total_Neuron_Count");
                }
                assign_entry(index, entries[index].fixed_count);
            }
        }

        const std::size_t remaining_budget = total_neurons - allocation.assigned_neurons;
        std::array<long double, max_neuron_mix_entries> remainders{};
        std::size_t percentage_assigned = 0U;
        for (std::size_t index = 0U; index < entry_count; ++index) {
            if (entries[index].mode != NeuronAllocationMode::percentage) {
                continue;
            }
            const long double exact =
                static_cast<long double>(remaining_budget) *
                static_cast<long double>(entries[index].fill_percent) / 100.0L;
            const std::size_t floor_count =
                static_cast<std::size_t>(std::floor(exact));
            remainders[index] = exact - static_cast<long double>(floor_count);
            assign_entry(index, floor_count);
            percentage_assigned += floor_count;
        }

        if (total_fill_percent() >= 100.0F - 1.0e-3F) {
            while (percentage_assigned < remaining_budget) {
                std::size_t best = entry_count;
                for (std::size_t index = 0U; index < entry_count; ++index) {
                    if (entries[index].mode == NeuronAllocationMode::percentage &&
                        (best == entry_count || remainders[index] > remainders[best])) {
                        best = index;
                    }
                }
                if (best == entry_count) {
                    throw std::invalid_argument(
                        "a complete neuron mix has no percentage entries");
                }
                remainders[best] = -1.0L;
                assign_entry(best, 1U);
                ++percentage_assigned;
            }
        }

        allocation.unassigned_neurons = total_neurons - allocation.assigned_neurons;
        return allocation;
    }
};

inline NeuronMixConfig default_neuron_mix() {
    NeuronMixConfig mix;
    mix.add_percentage(NeuronType::basic, 10.0F);
    mix.add_percentage(NeuronType::modern_gated, 35.0F);
    mix.add_percentage(NeuronType::compact_modern, 20.0F);
    mix.add_percentage(NeuronType::binary_inference, 10.0F);
    mix.add_percentage(NeuronType::sparse_modern, 25.0F);
    return mix;
}

}  // namespace agentari::neuron
