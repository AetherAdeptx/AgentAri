#include "agentari/HierarchicalPrediction.hpp"
#include "agentari/Parallel.hpp"

#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace agentari::text::hierarchical {
namespace {

constexpr std::uint32_t atomic_part_flag = std::uint32_t{1U} << 31U;
constexpr std::uint32_t atomic_part_mask = ~atomic_part_flag;

template <typename Value>
bool write_binary(std::ostream& output, const Value& value) {
    output.write(reinterpret_cast<const char*>(&value),
                 static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(output);
}

template <typename Value>
bool read_binary(std::istream& input, Value& value) {
    input.read(reinterpret_cast<char*>(&value),
               static_cast<std::streamsize>(sizeof(Value)));
    return static_cast<bool>(input);
}

float clamp_unit(const float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

}  // namespace

HierarchicalPredictionNetwork::HierarchicalPredictionNetwork(
    PredictionNetworkConfig config)
    : config_(config) {
    if (config_.neurons_per_layer == 0U || config_.tokenizer_neuron_count == 0U ||
        config_.min_neurons_per_layer == 0U ||
        config_.maximum_transition_neurons == 0U || config_.local_input_fraction < 0.0F ||
        config_.network_reference_fraction < 0.0F || config_.active_cross_layer_fraction < 0.0F ||
        config_.local_input_fraction > 1.0F || config_.network_reference_fraction > 1.0F ||
        config_.active_cross_layer_fraction > 1.0F || config_.learning_rate <= 0.0F ||
        config_.learning_rate > 1.0F ||
        config_.activation_decay < 0.0F || config_.activation_decay > 1.0F) {
        throw std::invalid_argument("invalid hierarchical prediction configuration");
    }
    if (std::any_of(config_.layer_size_weights.begin(), config_.layer_size_weights.end(),
                    [](const std::size_t weight) { return weight == 0U; })) {
        throw std::invalid_argument("every layer requires a non-zero size weight");
    }
    if (config_.additional_layer_size_weight == 0U) {
        throw std::invalid_argument("additional layers require a non-zero size weight");
    }
    if (config_.additional_layer_count >
        std::numeric_limits<std::size_t>::max() - prediction_layer_count) {
        throw std::invalid_argument("layer count overflows the configured size type");
    }
    if (config_.total_neuron_count == 0U) {
        if (config_.additional_layer_count >
            std::numeric_limits<std::size_t>::max() / config_.neurons_per_layer) {
            throw std::invalid_argument(
                "derived general-layer neuron count overflows the configured size type");
        }
        const std::size_t general_layer_count =
            config_.additional_layer_count * config_.neurons_per_layer;
        if (config_.tokenizer_neuron_count >
            std::numeric_limits<std::size_t>::max() - general_layer_count) {
            throw std::invalid_argument(
                "derived total neuron count overflows the configured size type");
        }
        config_.total_neuron_count = config_.tokenizer_neuron_count + general_layer_count;
    }
    if (config_.tokenizer_neuron_count <
        prediction_layer_count * config_.min_neurons_per_layer) {
        throw std::invalid_argument("tokenizer neuron count is too small for tokenizer layers");
    }
    if (config_.total_neuron_count < config_.tokenizer_neuron_count) {
        throw std::invalid_argument(
            "total neuron count is smaller than the fixed tokenizer neuron pool");
    }
    const std::size_t general_neuron_count =
        config_.total_neuron_count - config_.tokenizer_neuron_count;
    if (config_.additional_layer_count == 0U && general_neuron_count != 0U) {
        throw std::invalid_argument(
            "general-layer neurons require at least one additional layer");
    }
    if (config_.additional_layer_count != 0U &&
        config_.additional_layer_count >
            general_neuron_count / config_.min_neurons_per_layer) {
        throw std::invalid_argument(
            "general-layer neuron count is too small for layer minimums");
    }
    if (std::fabs((config_.local_input_fraction + config_.network_reference_fraction) - 1.0F) >
        1.0e-4F) {
        throw std::invalid_argument("local and outer fractions must sum to one");
    }

    const auto layer_counts = allocate_layer_counts();
    neuron_mix_allocation_ = config_.neuron_mix.allocate(
        config_.total_neuron_count,
        std::span<const std::size_t>(layer_counts.data(), layer_counts.size()));
    state_.total_neuron_count = config_.total_neuron_count;
    state_.tokenizer_neuron_count = config_.tokenizer_neuron_count;
    state_.layer_neuron_count = general_neuron_count;
    state_.layers.resize(layer_counts.size());
    neuron_groups_.resize(layer_counts.size());
    local_signal_scratch_.resize(layer_counts.size(), 0.0F);
    previous_activation_scratch_.resize(layer_counts.size(), 0.0F);
    for (std::size_t layer = 0U; layer < layer_counts.size(); ++layer) {
        initialize_layer_state(layer, layer_counts[layer]);
    }
    transitions_.fill(TransitionTable{});
    std::vector<SpatialNeuronAssignment> spatial_assignments;
    spatial_assignments.reserve(neuron_mix_allocation_.placements.size());
    for (const agentari::neuron::NeuronPlacement& placement :
         neuron_mix_allocation_.placements) {
        spatial_assignments.push_back(SpatialNeuronAssignment{
            .type_number = placement.number,
            .layer_id = placement.layer_id,
        });
    }
    spatial_map_.allocate(
        std::span<const std::size_t>(layer_counts.data(), layer_counts.size()),
        std::span<const SpatialNeuronAssignment>(spatial_assignments.data(),
                                                 spatial_assignments.size()));

    // The spatial map has already reserved every cell and assigned stable
    // type numbers. Construct each polymorphic neuron only after its cell is
    // known, preserving the allocation order used by the deterministic map.
    neuron_objects_.resize(spatial_map_.size());
    for (const SpatialNeuron& spatial_neuron : spatial_map_.neurons()) {
        if (spatial_neuron.type_number == 0U) {
            continue;
        }
        const std::size_t entry_index =
            static_cast<std::size_t>(spatial_neuron.type_number - 1U);
        if (entry_index >= config_.neuron_mix.entry_count) {
            throw std::logic_error("spatial neuron references an unknown mix entry");
        }
        const agentari::neuron::NeuronFactory& factory =
            config_.neuron_mix.entries[entry_index].factory;
        neuron_objects_[static_cast<std::size_t>(spatial_neuron.id)] =
            factory ? factory(spatial_neuron.id + 0xA17E'0000ULL)
                    : std::make_unique<agentari::neuron::Neuron>();
        if (neuron_objects_[static_cast<std::size_t>(spatial_neuron.id)] != nullptr) {
            ++instantiated_neuron_count_;
        }
    }
}

std::size_t HierarchicalPredictionNetwork::index(const PredictionLayer layer) noexcept {
    return static_cast<std::size_t>(layer);
}

std::vector<std::size_t>
HierarchicalPredictionNetwork::allocate_layer_counts() const {
    const std::size_t layer_count = prediction_layer_count + config_.additional_layer_count;
    std::vector<std::size_t> counts(layer_count, 0U);
    std::vector<long double> remainders(layer_count, 0.0L);

    const auto symmetric_rank = [](const std::size_t local_index,
                                   const std::size_t pool_size) noexcept {
        const std::size_t center = (pool_size - 1U) / 2U;
        if (local_index == center) {
            return std::size_t{0U};
        }
        const std::size_t distance = local_index > center
                                         ? local_index - center
                                         : center - local_index;
        // Prefer the positive side when an even-sized pool has two equally
        // central cells, then alternate deterministically outward.
        return distance * 2U + (local_index < center ? 1U : 0U);
    };

    const auto allocate_pool = [&](const std::size_t first_layer,
                                   const std::size_t pool_size,
                                   const std::size_t pool_budget,
                                   const auto& weight_for_layer) {
        if (pool_size == 0U) {
            if (pool_budget != 0U) {
                throw std::invalid_argument(
                    "a non-zero neuron pool requires layers to receive it");
            }
            return;
        }
        if (config_.min_neurons_per_layer >
            std::numeric_limits<std::size_t>::max() / pool_size) {
            throw std::invalid_argument("layer minimum total overflows the size type");
        }
        const std::size_t minimum_total = config_.min_neurons_per_layer * pool_size;
        if (pool_budget < minimum_total) {
            throw std::invalid_argument("neuron pool is too small for its layer minimums");
        }

        std::size_t weight_total = 0U;
        for (std::size_t local = 0U; local < pool_size; ++local) {
            const std::size_t weight = weight_for_layer(local);
            if (weight == 0U || weight_total >
                                    std::numeric_limits<std::size_t>::max() - weight) {
                throw std::invalid_argument(
                    "layer-size weights must be non-zero and fit the size type");
            }
            weight_total += weight;
        }
        if (weight_total == 0U) {
            throw std::invalid_argument("layer-size weights must have a non-zero total");
        }

        for (std::size_t local = 0U; local < pool_size; ++local) {
            counts[first_layer + local] = config_.min_neurons_per_layer;
        }
        const std::size_t remaining = pool_budget - minimum_total;
        std::size_t assigned = minimum_total;
        for (std::size_t local = 0U; local < pool_size; ++local) {
            const long double exact =
                static_cast<long double>(remaining) *
                static_cast<long double>(weight_for_layer(local)) /
                static_cast<long double>(weight_total);
            const std::size_t floor_count = static_cast<std::size_t>(std::floor(exact));
            counts[first_layer + local] += floor_count;
            remainders[first_layer + local] =
                exact - static_cast<long double>(floor_count);
            assigned += floor_count;
        }
        while (assigned < pool_budget) {
            std::size_t best_layer = first_layer;
            for (std::size_t local = 1U; local < pool_size; ++local) {
                const std::size_t candidate = first_layer + local;
                const std::size_t best_local = best_layer - first_layer;
                if (remainders[candidate] > remainders[best_layer] ||
                    (remainders[candidate] == remainders[best_layer] &&
                     symmetric_rank(local, pool_size) <
                         symmetric_rank(best_local, pool_size))) {
                    best_layer = candidate;
                }
            }
            ++counts[best_layer];
            remainders[best_layer] = -1.0L;
            ++assigned;
        }
    };

    allocate_pool(0U, prediction_layer_count, config_.tokenizer_neuron_count,
                  [this](const std::size_t local) {
                      return config_.layer_size_weights[local];
                  });
    allocate_pool(prediction_layer_count, config_.additional_layer_count,
                  config_.total_neuron_count - config_.tokenizer_neuron_count,
                  [this](const std::size_t) {
                      return config_.additional_layer_size_weight;
                  });

    const std::size_t minimum_per_layer = config_.min_neurons_per_layer;
    if (std::any_of(counts.begin(), counts.end(), [minimum_per_layer](const std::size_t count) {
            return count < minimum_per_layer;
        })) {
        throw std::invalid_argument("total neuron count is too small for layer minimums");
    }
    return counts;
}

void HierarchicalPredictionNetwork::initialize_layer_state(const std::size_t layer_index,
                                                            const std::size_t neuron_count) {
    const std::size_t local_count = static_cast<std::size_t>(
        std::floor(static_cast<float>(neuron_count) * config_.local_input_fraction));
    const std::size_t outer_count = neuron_count - local_count;
    const std::size_t calculated_cross_count = static_cast<std::size_t>(
        std::floor(static_cast<float>(outer_count) * config_.active_cross_layer_fraction));
    const std::size_t cross_count = config_.active_cross_layer_fraction > 0.0F
                                        ? std::max<std::size_t>(1U, calculated_cross_count)
                                        : 0U;
    state_.layers[layer_index] = PredictionLayerState{
        .neuron_count = neuron_count,
        .local_input_neurons = local_count,
        .network_reference_neurons = outer_count,
        .cross_layer_input_neurons = cross_count,
    };
    neuron_groups_[layer_index] = NeuronActivationGroups{};
}

std::uint32_t HierarchicalPredictionNetwork::part_symbol(const PartAnalysis& analysis,
                                                         const PartSpan& span) noexcept {
    if (span.learned) {
        return span.id;
    }
    if (span.begin >= analysis.source.size()) {
        return atomic_part_flag;
    }
    return atomic_part_flag |
           static_cast<std::uint32_t>(static_cast<unsigned char>(analysis.source[span.begin]));
}

void HierarchicalPredictionNetwork::route_neurons(
    const std::span<const float> local_signals,
    const std::size_t observation_count) {
    if (local_signals.size() != state_.layers.size() || observation_count == 0U) {
        return;
    }
    if (previous_activation_scratch_.size() != state_.layers.size()) {
        return;
    }
    std::vector<float>& previous_activations = previous_activation_scratch_;
    std::fill(previous_activations.begin(), previous_activations.end(), 0.0F);
    float total_activation = 0.0F;
    for (std::size_t layer = 0U; layer < state_.layers.size(); ++layer) {
        previous_activations[layer] = state_.layers[layer].activation;
        total_activation += previous_activations[layer];
    }

    for (std::size_t layer = 0U; layer < state_.layers.size(); ++layer) {
        const float other_sum = total_activation - previous_activations[layer];
        const std::size_t other_count = state_.layers.size() - 1U;
        const float outer_signal = other_count == 0U
                                       ? 0.0F
                                       : other_sum / static_cast<float>(other_count);
        PredictionLayerState& layer_state = state_.layers[layer];
        const float retained = config_.activation_decay;
        const float update = 1.0F - retained;
        NeuronActivationGroups& groups = neuron_groups_[layer];
        groups.local = clamp_unit(retained * groups.local + update * local_signals[layer]);
        groups.outer = clamp_unit(retained * groups.outer + update * outer_signal);
        if (layer_state.cross_layer_input_neurons > 0U) {
            // The cross-layer group receives the outer reference twice: once
            // as its normal outer input and once as the explicit feedback
            // handoff. It is clamped just as the former per-neuron loop was.
            groups.cross_layer = clamp_unit(retained * groups.cross_layer +
                                            update * clamp_unit(outer_signal + outer_signal));
        } else {
            groups.cross_layer = 0.0F;
        }
        const std::size_t outer_only_count =
            layer_state.network_reference_neurons - layer_state.cross_layer_input_neurons;
        const float activation_sum =
            static_cast<float>(layer_state.local_input_neurons) * groups.local +
            static_cast<float>(layer_state.cross_layer_input_neurons) * groups.cross_layer +
            static_cast<float>(outer_only_count) * groups.outer;
        layer_state.activation = layer_state.neuron_count == 0U
                                     ? 0.0F
                                     : activation_sum / static_cast<float>(layer_state.neuron_count);
        layer_state.network_reference_activation = outer_signal;
        layer_state.cross_layer_activation = groups.cross_layer;
        layer_state.observations += static_cast<std::uint64_t>(observation_count);
    }
}

void HierarchicalPredictionNetwork::add_local_signals(
    const HierarchicalTokenTrace& trace,
    const std::span<float> signals) const {
    if (signals.size() < prediction_layer_count) {
        return;
    }
    signals[index(PredictionLayer::character)] +=
        trace.atomic_tokens.empty() ? 0.0F : 1.0F;
    signals[index(PredictionLayer::word)] += trace.words.empty() ? 0.0F : 1.0F;
    signals[index(PredictionLayer::context)] +=
        trace.context.recent_words.empty() ? 0.0F : 1.0F;

    for (const WordTrace& word : trace.words) {
        for (const PartSpan& span : word.parts.spans) {
            if (span.learned) {
                signals[index(PredictionLayer::part)] += 1.0F;
                return;
            }
        }
    }
}

void HierarchicalPredictionNetwork::learn_transition(const PredictionLayer layer,
                                                     const std::uint32_t source,
                                                     const std::uint32_t target) {
    learn_transition_count(layer, source, target, 1U);
}

void HierarchicalPredictionNetwork::learn_transition_count(
    const PredictionLayer layer,
    const std::uint32_t source,
    const std::uint32_t target,
    const std::uint64_t count) {
    if (count == 0U) {
        return;
    }
    TransitionTable& table = transitions_[index(layer)];
    auto source_entry = table.find(source);
    if (source_entry == table.end()) {
        if (state_.layers[index(layer)].transition_neurons >=
            config_.maximum_transition_neurons) {
            return;
        }
        source_entry = table.emplace(source, std::unordered_map<std::uint32_t,
                                                                  AssociationNeuron>{})
                           .first;
    }
    auto& target_map = source_entry->second;
    auto target_entry = target_map.find(target);
    if (target_entry == target_map.end()) {
        if (state_.layers[index(layer)].transition_neurons >=
            config_.maximum_transition_neurons) {
            return;
        }
        target_entry = target_map.emplace(target, AssociationNeuron{}).first;
        ++state_.layers[index(layer)].transition_neurons;
    }
    AssociationNeuron& neuron = target_entry->second;
    const std::uint64_t available_count =
        std::numeric_limits<std::uint64_t>::max() - neuron.count;
    neuron.count += std::min(count, available_count);

    // Repeated online updates have a closed form. Applying that form preserves
    // the same bounded recurrence while avoiding one loop per repeated event
    // after worker-local deltas have been coalesced.
    const auto power = [](const float base, const std::uint64_t exponent) noexcept {
        if (exponent == 0U || base >= 1.0F) {
            return 1.0F;
        }
        if (base <= 0.0F) {
            return 0.0F;
        }
        return std::pow(base, static_cast<float>(exponent));
    };
    const float learning_retention = power(1.0F - config_.learning_rate, count);
    const float activation_retention = power(config_.activation_decay, count);
    neuron.weight = clamp_unit(1.0F - (1.0F - neuron.weight) * learning_retention);
    neuron.activation = clamp_unit(1.0F - (1.0F - neuron.activation) *
                                   activation_retention);
}

void HierarchicalPredictionNetwork::learn_transitions(const HierarchicalTokenTrace& trace) {
    // Every sequence gets explicit boundaries. This makes first-token
    // prediction possible and gives a future generator an unambiguous stop
    // signal without using an ASCII, part, or vocabulary slot.
    if (!trace.atomic_tokens.empty()) {
        learn_transition(PredictionLayer::character, sequence_begin_id,
                         trace.atomic_tokens.front().id);
        for (std::size_t position = 1U; position < trace.atomic_tokens.size(); ++position) {
            learn_transition(PredictionLayer::character,
                             trace.atomic_tokens[position - 1U].id,
                             trace.atomic_tokens[position].id);
        }
        learn_transition(PredictionLayer::character, trace.atomic_tokens.back().id,
                         sequence_end_id);
    }

    for (const WordTrace& word : trace.words) {
        if (word.parts.spans.empty()) {
            continue;
        }
        learn_transition(PredictionLayer::part, sequence_begin_id,
                         part_symbol(word.parts, word.parts.spans.front()));
        for (std::size_t position = 1U; position < word.parts.spans.size(); ++position) {
            learn_transition(PredictionLayer::part,
                             part_symbol(word.parts, word.parts.spans[position - 1U]),
                             part_symbol(word.parts, word.parts.spans[position]));
        }
        learn_transition(PredictionLayer::part,
                         part_symbol(word.parts, word.parts.spans.back()), sequence_end_id);
    }

    bool have_previous_word = false;
    WordId previous_word = WordVocabulary::unknown_id;
    for (const WordTrace& word : trace.words) {
        if (word.word == WordVocabulary::unknown_id) {
            continue;
        }
        if (!have_previous_word) {
            learn_transition(PredictionLayer::word, sequence_begin_id, word.word);
            have_previous_word = true;
        } else {
            learn_transition(PredictionLayer::word, previous_word, word.word);
            // The context layer is reverse-directed for now: current word -> left word.
            learn_transition(PredictionLayer::context, word.word, previous_word);
        }
        previous_word = word.word;
    }
    if (have_previous_word) {
        learn_transition(PredictionLayer::word, previous_word, sequence_end_id);
    }
}

void HierarchicalPredictionNetwork::collect_learning_delta(
    const HierarchicalTokenTrace& trace,
    PredictionLearningDelta& delta) const {
    delta.clear();
    append_learning_delta(trace, delta);
}

void HierarchicalPredictionNetwork::append_learning_delta(
    const HierarchicalTokenTrace& trace,
    PredictionLearningDelta& delta) const {
    const auto append = [&delta](const PredictionLayer layer,
                                 const std::uint32_t source,
                                 const std::uint32_t target) {
        delta.transitions.push_back(PredictionTransitionDelta{
            .layer = layer,
            .source = source,
            .target = target,
            .count = 1U,
        });
    };

    if (!trace.atomic_tokens.empty()) {
        append(PredictionLayer::character, sequence_begin_id,
               trace.atomic_tokens.front().id);
        for (std::size_t position = 1U; position < trace.atomic_tokens.size(); ++position) {
            append(PredictionLayer::character,
                   trace.atomic_tokens[position - 1U].id,
                   trace.atomic_tokens[position].id);
        }
        append(PredictionLayer::character, trace.atomic_tokens.back().id,
               sequence_end_id);
    }

    for (const WordTrace& word : trace.words) {
        if (word.parts.spans.empty()) {
            continue;
        }
        append(PredictionLayer::part, sequence_begin_id,
               part_symbol(word.parts, word.parts.spans.front()));
        for (std::size_t position = 1U; position < word.parts.spans.size(); ++position) {
            append(PredictionLayer::part,
                   part_symbol(word.parts, word.parts.spans[position - 1U]),
                   part_symbol(word.parts, word.parts.spans[position]));
        }
        append(PredictionLayer::part,
               part_symbol(word.parts, word.parts.spans.back()), sequence_end_id);
    }

    bool have_previous_word = false;
    WordId previous_word = WordVocabulary::unknown_id;
    for (const WordTrace& word : trace.words) {
        if (word.word == WordVocabulary::unknown_id) {
            continue;
        }
        if (!have_previous_word) {
            append(PredictionLayer::word, sequence_begin_id, word.word);
            have_previous_word = true;
        } else {
            append(PredictionLayer::word, previous_word, word.word);
            append(PredictionLayer::context, word.word, previous_word);
        }
        previous_word = word.word;
    }
    if (have_previous_word) {
        append(PredictionLayer::word, previous_word, sequence_end_id);
    }
}

void HierarchicalPredictionNetwork::coalesce_learning_delta(
    PredictionLearningDelta& delta) const {
    auto same_transition = [](const PredictionTransitionDelta& left,
                              const PredictionTransitionDelta& right) {
        return left.layer == right.layer && left.source == right.source &&
               left.target == right.target;
    };
    std::sort(delta.transitions.begin(), delta.transitions.end(),
              [](const PredictionTransitionDelta& left,
                 const PredictionTransitionDelta& right) {
                  if (left.layer != right.layer) {
                      return static_cast<std::uint8_t>(left.layer) <
                             static_cast<std::uint8_t>(right.layer);
                  }
                  if (left.source != right.source) {
                      return left.source < right.source;
                  }
                  return left.target < right.target;
              });

    std::size_t write = 0U;
    for (std::size_t read = 0U; read < delta.transitions.size(); ++read) {
        if (write != 0U && same_transition(delta.transitions[write - 1U],
                                            delta.transitions[read])) {
            const std::uint64_t count = delta.transitions[write - 1U].count;
            const std::uint64_t additional = delta.transitions[read].count;
            delta.transitions[write - 1U].count =
                count > std::numeric_limits<std::uint64_t>::max() - additional
                    ? std::numeric_limits<std::uint64_t>::max()
                    : count + additional;
        } else {
            delta.transitions[write++] = delta.transitions[read];
        }
    }
    delta.transitions.resize(write);
}

void HierarchicalPredictionNetwork::apply_learning_delta(
    const PredictionLearningDelta& delta) {
    for (const PredictionTransitionDelta& transition : delta.transitions) {
        if (transition.count == 0U) {
            continue;
        }
        learn_transition_count(transition.layer, transition.source, transition.target,
                               transition.count);
    }
}

void HierarchicalPredictionNetwork::observe(const HierarchicalTokenTrace& trace) {
    observe_batch(std::span<const HierarchicalTokenTrace>(&trace, 1U));
}

void HierarchicalPredictionNetwork::observe_batch(
    const std::span<const HierarchicalTokenTrace> traces) {
    if (traces.empty()) {
        return;
    }
    if (local_signal_scratch_.size() != state_.layers.size()) {
        return;
    }
    std::vector<float>& local_signals = local_signal_scratch_;
    std::fill(local_signals.begin(), local_signals.end(), 0.0F);
    for (const HierarchicalTokenTrace& trace : traces) {
        add_local_signals(trace,
                          std::span<float>(local_signals.data(), prediction_layer_count));
    }

    parallel::Config delta_config;
    delta_config.grain_size = 1U;
    delta_config.minimum_parallel_work = 32U;
    const std::size_t delta_workers =
        parallel::worker_count_for(traces.size(), delta_config);
    if (delta_workers <= 1U || traces.size() < 32U) {
        for (const HierarchicalTokenTrace& trace : traces) {
            learn_transitions(trace);
        }
    } else {
        if (learning_delta_scratch_.size() < delta_workers) {
            learning_delta_scratch_.resize(delta_workers);
        }
        parallel::Config worker_config = delta_config;
        worker_config.worker_count = delta_workers;
        worker_config.minimum_parallel_work = 1U;
        parallel::parallel_for(0U, delta_workers, [&](const std::size_t worker) {
            PredictionLearningDelta& delta = learning_delta_scratch_[worker];
            delta.clear();
            const std::size_t begin = traces.size() * worker / delta_workers;
            const std::size_t end = traces.size() * (worker + 1U) / delta_workers;
            for (std::size_t index = begin; index < end; ++index) {
                append_learning_delta(traces[index], delta);
            }
            coalesce_learning_delta(delta);
        }, worker_config);
        // Merge in worker-index order. This is deterministic and keeps all
        // shared transition-table mutations on the owning thread.
        for (std::size_t worker = 0U; worker < delta_workers; ++worker) {
            apply_learning_delta(learning_delta_scratch_[worker]);
        }
    }
    const float inverse_count = 1.0F / static_cast<float>(traces.size());
    for (std::size_t layer = 0U; layer < prediction_layer_count; ++layer) {
        local_signals[layer] *= inverse_count;
    }
    // General layers receive a local handoff from their immediate predecessor;
    // their 85% outer bank still references the rest of the complete network.
    for (std::size_t layer = prediction_layer_count; layer < local_signals.size(); ++layer) {
        local_signals[layer] = state_.layers[layer - 1U].activation;
    }
    route_neurons(std::span<const float>(local_signals.data(), local_signals.size()),
                  traces.size());
    ++state_.step;
}

void HierarchicalPredictionNetwork::adjust_teacher_transition(
    const PredictionLayer layer,
    const std::uint32_t source,
    const std::uint32_t target,
    const float quality,
    const float confidence) {
    if (!std::isfinite(quality) || !std::isfinite(confidence)) {
        return;
    }
    const float bounded_quality = clamp_unit(quality);
    const float bounded_confidence = clamp_unit(confidence);
    const float signal = (2.0F * bounded_quality - 1.0F) * bounded_confidence;
    auto source_entry = transitions_[index(layer)].find(source);
    if (source_entry == transitions_[index(layer)].end()) {
        return;
    }
    auto target_entry = source_entry->second.find(target);
    if (target_entry == source_entry->second.end()) {
        return;
    }
    AssociationNeuron& neuron = target_entry->second;
    // The bias is deliberately small per update and bounded. It changes the
    // ranking of an existing association without allowing an external judge
    // to allocate arbitrary transitions or dominate frequency learning.
    neuron.teacher_bias = std::clamp(neuron.teacher_bias + 0.25F * signal,
                                     -4.0F, 4.0F);
}

void HierarchicalPredictionNetwork::apply_teacher_feedback(
    const HierarchicalTokenTrace& trace,
    const float quality,
    const float confidence) {
    if (!std::isfinite(quality) || !std::isfinite(confidence)) {
        return;
    }
    const float bounded_quality = clamp_unit(quality);
    const float bounded_confidence = clamp_unit(confidence);

    auto adjust_sequence = [this, bounded_quality, bounded_confidence](
                               const PredictionLayer layer,
                               const std::span<const std::uint32_t> ids) {
        if (ids.empty()) {
            return;
        }
        adjust_teacher_transition(layer, sequence_begin_id, ids.front(),
                                  bounded_quality, bounded_confidence);
        for (std::size_t position = 1U; position < ids.size(); ++position) {
            adjust_teacher_transition(layer, ids[position - 1U], ids[position],
                                      bounded_quality, bounded_confidence);
        }
        adjust_teacher_transition(layer, ids.back(), sequence_end_id,
                                  bounded_quality, bounded_confidence);
    };

    atomic_id_scratch_.clear();
    atomic_id_scratch_.reserve(trace.atomic_tokens.size());
    for (const AtomicToken& token : trace.atomic_tokens) {
        atomic_id_scratch_.push_back(token.id);
    }
    adjust_sequence(PredictionLayer::character,
                    std::span<const std::uint32_t>(atomic_id_scratch_.data(),
                                                   atomic_id_scratch_.size()));

    for (const WordTrace& word : trace.words) {
        part_id_scratch_.clear();
        part_id_scratch_.reserve(word.parts.spans.size());
        for (const PartSpan& span : word.parts.spans) {
            part_id_scratch_.push_back(part_symbol(word.parts, span));
        }
        adjust_sequence(PredictionLayer::part,
                        std::span<const std::uint32_t>(part_id_scratch_.data(),
                                                       part_id_scratch_.size()));
    }

    known_word_scratch_.clear();
    known_word_scratch_.reserve(trace.words.size());
    for (const WordTrace& word : trace.words) {
        if (word.word != WordVocabulary::unknown_id) {
            known_word_scratch_.push_back(word.word);
        }
    }
    adjust_sequence(PredictionLayer::word,
                    std::span<const std::uint32_t>(known_word_scratch_.data(),
                                                   known_word_scratch_.size()));
    for (std::size_t position = 1U; position < known_word_scratch_.size(); ++position) {
        // Context is reverse-directed: the current word points to the word
        // immediately to its left.
        adjust_teacher_transition(PredictionLayer::context,
                                  known_word_scratch_[position],
                                  known_word_scratch_[position - 1U], bounded_quality,
                                  bounded_confidence);
    }
}

HierarchicalTokenTrace HierarchicalPredictionNetwork::train_step(
    HierarchicalTokenizer& tokenizer, const std::string_view input) {
    HierarchicalTokenTrace trace = tokenizer.observe(input);
    observe(trace);
    return trace;
}

void HierarchicalPredictionNetwork::observe(HierarchicalTokenizer& tokenizer,
                                            const std::string_view input) {
    (void)train_step(tokenizer, input);
}

std::vector<PredictionCandidate> HierarchicalPredictionNetwork::predict(
    const PredictionLayer layer, const std::uint32_t source, const std::size_t top_k) const {
    if (top_k == 0U) {
        return {};
    }
    const TransitionTable& table = transitions_[index(layer)];
    const auto source_entry = table.find(source);
    if (source_entry == table.end()) {
        return {};
    }
    std::vector<PredictionCandidate> result;
    result.reserve(source_entry->second.size());
    float total_score = 0.0F;
    for (const auto& [target, neuron] : source_entry->second) {
        // Counts carry the learned frequency signal. Weight and activation
        // remain small adaptive signals, but no longer drown out frequency
        // when many associations have saturated their bounded weight.
        const std::uint64_t bounded_count =
            std::min<std::uint64_t>(neuron.count, 1'000'000'000ULL);
        const float base_score = static_cast<float>(bounded_count) +
                                 0.01F * neuron.weight + 0.01F * neuron.activation;
        const float score = base_score *
                            std::exp(std::clamp(neuron.teacher_bias, -4.0F, 4.0F));
        const bool is_end = target == sequence_end_id;
        const bool is_atomic_part = layer == PredictionLayer::part &&
                                     !is_end && (target & atomic_part_flag) != 0U;
        result.push_back(PredictionCandidate{
            .id = is_atomic_part ? target & atomic_part_mask : target,
            .count = neuron.count,
            .probability = score,
            .atomic_fallback = is_atomic_part,
            .sequence_end = is_end,
        });
        total_score += score;
    }
    std::sort(result.begin(), result.end(), [](const PredictionCandidate& left,
                                               const PredictionCandidate& right) {
        if (left.probability != right.probability) {
            return left.probability > right.probability;
        }
        return left.id < right.id;
    });
    if (result.size() > top_k) {
        result.resize(top_k);
    }
    for (PredictionCandidate& candidate : result) {
        candidate.probability = total_score == 0.0F
                                    ? 0.0F
                                    : candidate.probability / total_score;
    }
    return result;
}

std::vector<PredictionCandidate> HierarchicalPredictionNetwork::predict_next_character(
    const std::uint32_t current, const std::size_t top_k) const {
    return predict(PredictionLayer::character, current, top_k);
}

std::vector<PredictionCandidate> HierarchicalPredictionNetwork::predict_next_part(
    const PartId current, const std::size_t top_k) const {
    return predict(PredictionLayer::part, current, top_k);
}

std::vector<PredictionCandidate> HierarchicalPredictionNetwork::predict_next_word(
    const WordId current, const std::size_t top_k) const {
    return predict(PredictionLayer::word, current, top_k);
}

std::vector<PredictionCandidate> HierarchicalPredictionNetwork::predict_left_context(
    const WordId current, const std::size_t top_k) const {
    return predict(PredictionLayer::context, current, top_k);
}

std::optional<PredictionAssociationSnapshot>
HierarchicalPredictionNetwork::inspect_association(const PredictionLayer layer,
                                                    const std::uint32_t source,
                                                    const std::uint32_t target) const {
    const auto& table = transitions_[index(layer)];
    const auto source_entry = table.find(source);
    if (source_entry == table.end()) {
        return std::nullopt;
    }
    const auto target_entry = source_entry->second.find(target);
    if (target_entry == source_entry->second.end()) {
        return std::nullopt;
    }
    const AssociationNeuron& neuron = target_entry->second;
    return PredictionAssociationSnapshot{
        .count = neuron.count,
        .weight = neuron.weight,
        .activation = neuron.activation,
        .teacher_bias = neuron.teacher_bias,
    };
}

const PredictionNetworkConfig& HierarchicalPredictionNetwork::config() const noexcept {
    return config_;
}

const agentari::neuron::NeuronMixConfig& HierarchicalPredictionNetwork::neuron_mix()
    const noexcept {
    return config_.neuron_mix;
}

const agentari::neuron::NeuronMixAllocation&
HierarchicalPredictionNetwork::neuron_mix_allocation() const noexcept {
    return neuron_mix_allocation_;
}

const std::vector<std::unique_ptr<agentari::neuron::Neuron>>&
HierarchicalPredictionNetwork::neurons() const noexcept {
    return neuron_objects_;
}

std::size_t HierarchicalPredictionNetwork::instantiated_neuron_count() const noexcept {
    return instantiated_neuron_count_;
}

PredictionNetworkState HierarchicalPredictionNetwork::state() const {
    return state_;
}

const SpatialNeuronMap& HierarchicalPredictionNetwork::spatial_map() const noexcept {
    return spatial_map_;
}

std::size_t HierarchicalPredictionNetwork::layer_count() const noexcept {
    return state_.layers.size();
}

bool HierarchicalPredictionNetwork::save(std::ostream& output, std::string& error) const {
    constexpr std::array<char, 8U> magic{{'A', 'R', 'I', 'P', 'R', 'D', '0', '2'}};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!output || !write_binary(output, state_.step)) {
        error = "could not write prediction state header";
        return false;
    }
    for (std::size_t layer = 0U; layer < prediction_layer_count; ++layer) {
        const std::uint64_t source_count = transitions_[layer].size();
        if (!write_binary(output, source_count)) {
            error = "could not write prediction source count";
            return false;
        }
        for (const auto& [source, targets] : transitions_[layer]) {
            if (!write_binary(output, source)) {
                error = "could not write prediction source";
                return false;
            }
            const std::uint64_t target_count = targets.size();
            if (!write_binary(output, target_count)) {
                error = "could not write prediction target count";
                return false;
            }
            for (const auto& [target, neuron] : targets) {
                if (!write_binary(output, target) || !write_binary(output, neuron.count) ||
                    !write_binary(output, neuron.weight) ||
                    !write_binary(output, neuron.activation) ||
                    !write_binary(output, neuron.teacher_bias)) {
                    error = "could not write prediction neuron";
                    return false;
                }
            }
        }
    }
    return static_cast<bool>(output);
}

bool HierarchicalPredictionNetwork::load(std::istream& input, std::string& error) {
    constexpr std::array<char, 8U> magic{{'A', 'R', 'I', 'P', 'R', 'D', '0', '2'}};
    constexpr std::array<char, 8U> legacy_magic{{'A', 'R', 'I', 'P', 'R', 'D', '0', '1'}};
    std::array<char, 8U> stored_magic{};
    if (!input.read(stored_magic.data(), static_cast<std::streamsize>(stored_magic.size())) ||
        (stored_magic != magic && stored_magic != legacy_magic) ||
        !read_binary(input, state_.step)) {
        error = "invalid prediction state header";
        return false;
    }
    const bool legacy_format = stored_magic == legacy_magic;
    for (TransitionTable& table : transitions_) {
        table.clear();
    }
    for (PredictionLayerState& layer : state_.layers) {
        layer.transition_neurons = 0U;
    }
    for (std::size_t layer = 0U; layer < prediction_layer_count; ++layer) {
        std::uint64_t source_count = 0U;
        if (!read_binary(input, source_count) ||
            source_count > config_.maximum_transition_neurons) {
            error = "invalid prediction source count";
            return false;
        }
        for (std::uint64_t source_index = 0U; source_index < source_count; ++source_index) {
            std::uint32_t source = 0U;
            std::uint64_t target_count = 0U;
            if (!read_binary(input, source) || !read_binary(input, target_count) ||
                target_count > config_.maximum_transition_neurons) {
                error = "invalid prediction target count";
                return false;
            }
            auto& targets = transitions_[layer][source];
            for (std::uint64_t target_index = 0U; target_index < target_count; ++target_index) {
                std::uint32_t target = 0U;
                AssociationNeuron neuron{};
                if (!read_binary(input, target) || !read_binary(input, neuron.count) ||
                    !read_binary(input, neuron.weight) ||
                    !read_binary(input, neuron.activation) ||
                    (!legacy_format && !read_binary(input, neuron.teacher_bias))) {
                    error = "invalid prediction neuron";
                    return false;
                }
                if (legacy_format) {
                    neuron.teacher_bias = 0.0F;
                }
                if (state_.layers[layer].transition_neurons >=
                    config_.maximum_transition_neurons) {
                    error = "prediction transition capacity exceeded";
                    return false;
                }
                targets.emplace(target, neuron);
                ++state_.layers[layer].transition_neurons;
            }
        }
    }
    return true;
}

}  // namespace agentari::text::hierarchical
