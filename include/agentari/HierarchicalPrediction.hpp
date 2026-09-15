#pragma once

#include "agentari/SpatialNeuronMap.hpp"
#include "agentari/HierarchicalTokenizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace agentari::text::hierarchical {

enum class PredictionLayer : std::uint8_t {
    character = 0U,
    part = 1U,
    word = 2U,
    context = 3U,
};

inline constexpr std::size_t prediction_layer_count = 4U;
inline constexpr std::size_t tokenizer_prediction_layer_count = prediction_layer_count;
// These IDs are reserved as sequence boundaries. They are kept outside the
// current character, part, and word ranges so a model can learn how a sample
// begins and ends without consuming an ASCII slot or a vocabulary slot.
inline constexpr std::uint32_t sequence_begin_id =
    std::numeric_limits<std::uint32_t>::max() - 1U;
inline constexpr std::uint32_t sequence_end_id =
    std::numeric_limits<std::uint32_t>::max();
// This is the minimum total neuron bank assigned to every prediction layer.
// It is a named project-wide default so experiments can override it in one
// place or per PredictionNetworkConfig.
inline constexpr std::size_t Min_Neurons_Per_Layer = 1U;

struct PredictionNetworkConfig {
    // A zero total derives the budget from the layer count and per-layer size.
    // The default is 4 tokenizer-prediction layers plus 24 general layers,
    // each with 2,000 neurons.
    std::size_t total_neuron_count{0U};
    std::size_t neurons_per_layer{2000U};
    std::size_t additional_layer_count{24U};
    std::array<std::size_t, prediction_layer_count> layer_size_weights{
        1U, 1U, 1U, 1U};
    std::size_t additional_layer_size_weight{1U};
    std::size_t min_neurons_per_layer{Min_Neurons_Per_Layer};

    // The first routing split is deliberate: 15% of a layer bank is local to
    // that layer, while 85% belongs to the outer/network-reference bank. For
    // now only a 10% subset of that outer bank receives active feedback from
    // other tokenizer layers.
    float local_input_fraction{0.15F};
    float network_reference_fraction{0.85F};
    float active_cross_layer_fraction{0.10F};
    float learning_rate{0.08F};
    float activation_decay{0.90F};
    std::size_t maximum_transition_neurons{262144U};
};

struct PredictionCandidate {
    std::uint32_t id{0U};
    std::uint64_t count{0U};
    float probability{0.0F};
    // Part predictions may return an atomic character fallback. Character and
    // word predictions always return false here.
    bool atomic_fallback{false};
    // An end boundary is returned after the final learned token when the
    // observed sequence ended there. This lets generation stop cleanly.
    bool sequence_end{false};
};

struct PredictionLayerState {
    std::size_t neuron_count{0U};
    std::size_t local_input_neurons{0U};
    std::size_t network_reference_neurons{0U};
    std::size_t cross_layer_input_neurons{0U};
    std::uint64_t observations{0U};
    std::uint64_t transition_neurons{0U};
    float activation{0.0F};
    float network_reference_activation{0.0F};
    float cross_layer_activation{0.0F};
};

struct PredictionNetworkState {
    std::uint64_t step{0U};
    std::size_t total_neuron_count{0U};
    std::vector<PredictionLayerState> layers{};
};

struct PredictionAssociationSnapshot {
    std::uint64_t count{0U};
    float weight{0.0F};
    float activation{0.0F};
    float teacher_bias{0.0F};
};

struct PredictionTransitionDelta {
    PredictionLayer layer{PredictionLayer::character};
    std::uint32_t source{0U};
    std::uint32_t target{0U};
    std::uint64_t count{1U};
};

struct PredictionLearningDelta {
    std::vector<PredictionTransitionDelta> transitions;

    void clear() noexcept { transitions.clear(); }
};

class HierarchicalPredictionNetwork {
public:
    explicit HierarchicalPredictionNetwork(PredictionNetworkConfig config = {});

    // Performs one online learning step: tokenize and update the tokenizer,
    // route activation through every layer, and learn sparse transitions in
    // the same call. No checkpoint or delayed training phase is required.
    [[nodiscard]] HierarchicalTokenTrace train_step(HierarchicalTokenizer& tokenizer,
                                                    std::string_view input);
    void observe(const HierarchicalTokenTrace& trace);
    // Learns all transitions in the supplied records but routes the dense
    // layer state once for the batch. This keeps record boundaries while
    // avoiding a full network pass for every input line.
    void observe_batch(std::span<const HierarchicalTokenTrace> traces);
    // Collects read-only transition evidence into a caller-owned delta. The
    // delta can be built by a worker and merged later by the owning network.
    void collect_learning_delta(const HierarchicalTokenTrace& trace,
                                PredictionLearningDelta& delta) const;
    void apply_learning_delta(const PredictionLearningDelta& delta);
    // Adjusts existing associations using a bounded external-teacher signal;
    // it never creates new transitions from feedback alone.
    void apply_teacher_feedback(const HierarchicalTokenTrace& trace,
                                float quality,
                                float confidence = 1.0F);
    void observe(HierarchicalTokenizer& tokenizer, std::string_view input);

    [[nodiscard]] std::vector<PredictionCandidate> predict_next_character(
        std::uint32_t current, std::size_t top_k = 5U) const;
    [[nodiscard]] std::vector<PredictionCandidate> predict_next_part(
        PartId current, std::size_t top_k = 5U) const;
    [[nodiscard]] std::vector<PredictionCandidate> predict_next_word(
        WordId current, std::size_t top_k = 5U) const;
    [[nodiscard]] std::vector<PredictionCandidate> predict_left_context(
        WordId current, std::size_t top_k = 5U) const;

    // Read-only diagnostics for one learned association neuron. A missing
    // value means that transition has not been allocated yet.
    [[nodiscard]] std::optional<PredictionAssociationSnapshot>
    inspect_association(PredictionLayer layer,
                        std::uint32_t source,
                        std::uint32_t target) const;

    [[nodiscard]] const PredictionNetworkConfig& config() const noexcept;
    [[nodiscard]] PredictionNetworkState state() const;
    [[nodiscard]] const SpatialNeuronMap& spatial_map() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;

    // Serialization hooks remain available for a future experiment, but the
    // current executable deliberately does not call them: learning is fresh
    // and in-memory for every run.
    [[nodiscard]] bool save(std::ostream& output, std::string& error) const;
    [[nodiscard]] bool load(std::istream& input, std::string& error);

private:
    struct AssociationNeuron {
        std::uint64_t count{0U};
        float weight{0.0F};
        float activation{0.0F};
        float teacher_bias{0.0F};
    };

    using TransitionTable =
        std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, AssociationNeuron>>;

    struct NeuronActivationGroups {
        float local{0.0F};
        float cross_layer{0.0F};
        float outer{0.0F};
    };

    PredictionNetworkConfig config_;
    PredictionNetworkState state_;
    SpatialNeuronMap spatial_map_;
    // Every neuron in one routing group receives the same scalar input. Keep
    // one representative activation per group instead of touching tens of
    // thousands of identical floats on every training record.
    std::vector<NeuronActivationGroups> neuron_groups_;
    // Reused process-lifetime scratch buffers. The online learner owns one
    // network, so these buffers are deliberately not thread-safe; they avoid
    // repeated heap activity without pretending the mutable learner is
    // concurrently writable.
    std::vector<float> local_signal_scratch_;
    std::vector<float> previous_activation_scratch_;
    std::vector<std::uint32_t> atomic_id_scratch_;
    std::vector<std::uint32_t> part_id_scratch_;
    std::vector<WordId> known_word_scratch_;
    // One reusable delta buffer per active batch worker. Deltas are never
    // shared between workers; only the owning thread mutates transitions_.
    std::vector<PredictionLearningDelta> learning_delta_scratch_;
    std::array<TransitionTable, prediction_layer_count> transitions_;

    [[nodiscard]] static std::size_t index(PredictionLayer layer) noexcept;
    [[nodiscard]] std::vector<std::size_t>
    allocate_layer_counts() const;
    [[nodiscard]] static std::uint32_t part_symbol(const PartAnalysis& analysis,
                                                    const PartSpan& span) noexcept;
    void route_neurons(std::span<const float> local_signals,
                       std::size_t observation_count);
    void add_local_signals(const HierarchicalTokenTrace& trace,
                           std::span<float> signals) const;
    void learn_transitions(const HierarchicalTokenTrace& trace);
    void append_learning_delta(const HierarchicalTokenTrace& trace,
                               PredictionLearningDelta& delta) const;
    void coalesce_learning_delta(PredictionLearningDelta& delta) const;
    void adjust_teacher_transition(PredictionLayer layer,
                                   std::uint32_t source,
                                   std::uint32_t target,
                                   float quality,
                                   float confidence);
    void learn_transition(PredictionLayer layer, std::uint32_t source, std::uint32_t target);
    void learn_transition_count(PredictionLayer layer,
                                std::uint32_t source,
                                std::uint32_t target,
                                std::uint64_t count);
    [[nodiscard]] std::vector<PredictionCandidate> predict(PredictionLayer layer,
                                                             std::uint32_t source,
                                                             std::size_t top_k) const;
    void initialize_layer_state(std::size_t layer_index, std::size_t neuron_count);
};

}  // namespace agentari::text::hierarchical
