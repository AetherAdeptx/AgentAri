#include "agentari/HierarchicalPrediction.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/TeacherFeedback.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void predictive_layers_learn() {
    using namespace agentari::text::hierarchical;

    HierarchicalTokenizer tokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 64U,
            .minimum_part_length = 2U,
            .maximum_part_length = 8U,
            .minimum_distinct_words = 2U,
            .context_radius = 2U,
        },
        .vocabulary_capacity = 128U,
        .context = ContextTokenizerConfig{.recent_word_limit = 16U},
    });
    HierarchicalPredictionNetwork network(PredictionNetworkConfig{
        .total_neuron_count = 56000U,
        .neurons_per_layer = 2000U,
        .additional_layer_count = 24U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 4096U,
    });

    for (std::size_t repeat = 0U; repeat < 4U; ++repeat) {
        network.observe(tokenizer, "abcabc");
        network.observe(tokenizer, "the quick");
        network.observe(tokenizer, "the quick");
        network.observe(tokenizer, "a b c");
    }
    const HierarchicalTokenTrace online_trace = network.train_step(tokenizer, "abcabc");
    require(!online_trace.atomic_tokens.empty(),
            "online training step did not return its token trace");
    // Establish repeated, context-bearing word families before training the
    // part-transition layer.
    network.observe(tokenizer, "unhappy happyish");
    network.observe(tokenizer, "unkind darkness");
    const auto part_trace = tokenizer.observe("unhappy");
    network.observe(part_trace);

    const auto next_letter = network.predict_next_character(static_cast<AtomicId>('a'), 1U);
    require(!next_letter.empty() && next_letter.front().id == static_cast<std::uint32_t>('b'),
            "character predictor did not learn the next letter");
    const auto first_letters = network.predict_next_character(sequence_begin_id, 64U);
    bool learned_first_letter = false;
    for (const PredictionCandidate& candidate : first_letters) {
        learned_first_letter = learned_first_letter || candidate.id == static_cast<std::uint32_t>('t');
    }
    require(learned_first_letter, "character predictor did not learn a sequence start");

    const PartId un = tokenizer.parts().lookup("un");
    const PartId happy = tokenizer.parts().lookup("happy");
    require(un != 0U && happy != 0U, "test word parts were not learned");
    const auto next_part = network.predict_next_part(un, 8U);
    bool found_happy = false;
    for (const PredictionCandidate& candidate : next_part) {
        found_happy = found_happy || (!candidate.atomic_fallback && candidate.id == happy);
    }
    require(found_happy, "part predictor did not learn the next part");

    const WordId the = tokenizer.words().lookup("the");
    const WordId quick = tokenizer.words().lookup("quick");
    require(the != WordVocabulary::unknown_id && quick != WordVocabulary::unknown_id,
            "test words were not learned");
    const auto next_word = network.predict_next_word(the, 1U);
    require(!next_word.empty() && next_word.front().id == quick,
            "word predictor did not learn the next word");
    const auto first_words = network.predict_next_word(sequence_begin_id, 64U);
    bool learned_first_word = false;
    for (const PredictionCandidate& candidate : first_words) {
        learned_first_word = learned_first_word || candidate.id == the;
    }
    require(learned_first_word, "word predictor did not learn a sequence start");
    const auto word_end = network.predict_next_word(quick, 8U);
    bool learned_word_end = false;
    for (const PredictionCandidate& candidate : word_end) {
        learned_word_end = learned_word_end || candidate.sequence_end;
    }
    require(learned_word_end, "word predictor did not learn a sequence end");

    const WordId c = tokenizer.words().lookup("c");
    const WordId b = tokenizer.words().lookup("b");
    const auto left_context = network.predict_left_context(c, 1U);
    require(!left_context.empty() && left_context.front().id == b,
            "context predictor did not learn the word on the left");

    const PredictionNetworkState state = network.state();
    require(state.step > 0U, "prediction network did not advance its state");
    require(network.layer_count() == prediction_layer_count + 24U &&
                state.layers.size() == prediction_layer_count + 24U,
            "prediction network did not create the tokenizer plus 24 additional layers");
    std::size_t total_neurons = 0U;
    for (const PredictionLayerState& layer : state.layers) {
        total_neurons += layer.neuron_count;
        require(layer.neuron_count == 2000U,
                "default additional layers did not receive 2,000 neurons");
        require(layer.local_input_neurons + layer.network_reference_neurons ==
                    layer.neuron_count,
                "prediction routing did not partition the neuron bank");
        require(layer.network_reference_neurons > layer.local_input_neurons,
                "prediction routing did not reserve the 85 percent outer bank");
        require(layer.cross_layer_input_neurons > 0U,
                "prediction routing did not activate cross-layer inputs");
    }
    require(total_neurons == state.total_neuron_count && total_neurons == 56000U,
            "layer allocations did not consume the global neuron budget");

    const SpatialNeuronMap& spatial = network.spatial_map();
    require(spatial.size() == total_neurons, "spatial map missed allocated neurons");
    require(spatial.layers().size() == prediction_layer_count + 24U,
            "spatial map did not create one region per layer");
    for (const SpatialLayerRegion& region : spatial.layers()) {
        require(region.neuron_count == 2000U && region.width > 0U && region.height > 0U &&
                    region.depth > 0U && region.width <= spatial_grid_width &&
                    region.height <= spatial_grid_width && region.depth <= spatial_grid_width,
                "spatial layer region has invalid geometry");
        require(region.anchor.x < spatial_grid_width && region.anchor.y < spatial_grid_width &&
                    region.anchor.z < spatial_grid_width,
                "spatial layer anchor escaped the grid");
        require(region.position.coordinate() == region.anchor && region.position.color() == 0U,
                "packed layer position does not match its 3D anchor");
    }
    require(spatial.occupied_cell_count() > 0U &&
                spatial.occupied_cell_count() <= spatial.size(),
            "spatial map occupancy index is invalid");
    std::vector<std::size_t> mapped_layer_counts(spatial.layer_count(), 0U);
    for (std::size_t neuron_index = 0U; neuron_index < spatial.neurons().size();
         ++neuron_index) {
        const SpatialNeuron& neuron = spatial.neurons()[neuron_index];
        require(neuron.id == neuron_index && neuron.layer_id < spatial.layer_count() &&
                    neuron.layer_local_index < spatial.layers()[neuron.layer_id].neuron_count,
                "spatial neuron was assigned to the wrong layer or local slot");
        require(neuron.coordinate.x < spatial_grid_width &&
                    neuron.coordinate.y < spatial_grid_width &&
                    neuron.coordinate.z < spatial_grid_width,
                "spatial neuron escaped the 128 cubed grid");
        require(neuron.position.coordinate() == neuron.coordinate && neuron.position.color() == 0U,
                "packed neuron position does not match its 3D coordinate");
        ++mapped_layer_counts[neuron.layer_id];
        require(spatial.find(neuron.id) != nullptr,
                "spatial map ID lookup failed");
        require(spatial.find(neuron.coordinate) != nullptr,
                "spatial map coordinate lookup failed");
    }
    for (const std::size_t count : mapped_layer_counts) {
        require(count == 2000U, "spatial map did not retain every neuron in its layer");
    }

    const SpatialNeuron& first_neuron = spatial.neurons().front();
    const SpatialSearchQuery exact_query{
        .center = first_neuron.coordinate,
        .radius = 0U,
        .layer_id = first_neuron.layer_id,
        .max_results = 8U,
    };
    const std::vector<SpatialNeuron> exact_matches = spatial.smart_search(exact_query);
    require(!exact_matches.empty() && exact_matches.front().coordinate == first_neuron.coordinate,
            "smart spatial search missed the exact coordinate");

    std::size_t lazy_matches = 0U;
    const std::size_t visited = spatial.lazy_search(
        SpatialSearchQuery{
            .center = first_neuron.coordinate,
            .radius = 4U,
            .layer_id = first_neuron.layer_id,
            .max_results = 3U,
        },
        [&lazy_matches](const SpatialNeuron&) {
            ++lazy_matches;
            return true;
        });
    require(visited == lazy_matches && visited <= 3U,
            "lazy spatial search did not honor its result bound");

    HierarchicalPredictionNetwork scaled_network(PredictionNetworkConfig{
        .total_neuron_count = 10000U,
        .neurons_per_layer = 1U,
        .additional_layer_count = 0U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 4096U,
    });
    std::size_t scaled_total = 0U;
    for (const PredictionLayerState& layer : scaled_network.state().layers) {
        scaled_total += layer.neuron_count;
        require(layer.neuron_count >= 16U,
                "global minimum was not preserved during rescaling");
    }
    require(scaled_total == 10000U && scaled_network.layer_count() == prediction_layer_count,
            "global budget did not rescale correctly when extra layers were removed");
}

void batched_observation_contract() {
    using namespace agentari::text::hierarchical;

    HierarchicalTokenizer tokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 32U,
            .minimum_part_length = 2U,
            .maximum_part_length = 8U,
            .minimum_distinct_words = 2U,
        },
        .vocabulary_capacity = 64U,
    });
    HierarchicalPredictionNetwork network(PredictionNetworkConfig{
        .total_neuron_count = 4000U,
        .neurons_per_layer = 1000U,
        .additional_layer_count = 0U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 512U,
    });

    std::array<HierarchicalTokenTrace, 2U> traces{
        tokenizer.observe("alpha beta"),
        tokenizer.observe("alpha gamma"),
    };
    network.observe_batch(std::span<const HierarchicalTokenTrace>(traces));

    const WordId alpha = tokenizer.words().lookup("alpha");
    const WordId beta = tokenizer.words().lookup("beta");
    const WordId gamma = tokenizer.words().lookup("gamma");
    const auto predictions = network.predict_next_word(alpha, 4U);
    bool found_beta = false;
    bool found_gamma = false;
    for (const PredictionCandidate& candidate : predictions) {
        found_beta = found_beta || candidate.id == beta;
        found_gamma = found_gamma || candidate.id == gamma;
    }
    require(found_beta && found_gamma, "batched observation lost a word transition");
    require(network.state().step == 1U, "batched observation performed multiple route passes");
}

void worker_local_delta_contract() {
    using namespace agentari::text::hierarchical;

    HierarchicalTokenizer tokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 32U,
            .minimum_part_length = 2U,
            .maximum_part_length = 8U,
            .minimum_distinct_words = 2U,
        },
        .vocabulary_capacity = 64U,
    });
    HierarchicalPredictionNetwork network(PredictionNetworkConfig{
        .total_neuron_count = 4000U,
        .neurons_per_layer = 1000U,
        .additional_layer_count = 0U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 512U,
    });

    agentari::parallel::set_default_worker_count(2U);
    std::vector<HierarchicalTokenTrace> traces;
    traces.reserve(64U);
    for (std::size_t index = 0U; index < 64U; ++index) {
        traces.push_back(tokenizer.observe("alpha beta"));
    }
    require(traces.front().words.size() == 2U && traces.front().words[0U].word == 1U &&
                traces.front().words[1U].word == 2U,
            "delta test did not retain the expected two known words");
    PredictionLearningDelta direct_delta;
    network.collect_learning_delta(traces.front(), direct_delta);
    std::size_t direct_word_transitions = 0U;
    for (const PredictionTransitionDelta& transition : direct_delta.transitions) {
        direct_word_transitions += transition.layer == PredictionLayer::word &&
                                           transition.source == 1U &&
                                           transition.target == 2U
                                       ? 1U
                                       : 0U;
    }
    require(direct_word_transitions == 1U,
            "delta collector did not retain the expected word transition");
    network.observe_batch(std::span<const HierarchicalTokenTrace>(traces));
    agentari::parallel::set_default_worker_count(0U);

    const WordId alpha = tokenizer.words().lookup("alpha");
    const WordId beta = tokenizer.words().lookup("beta");
    const auto association = network.inspect_association(PredictionLayer::word, alpha, beta);
    require(association.has_value() && association->count == 64U,
            "worker-local learning deltas were not merged exactly once: count=" +
                std::to_string(association.has_value() ? association->count : 0U) +
                " alpha=" + std::to_string(alpha) + " beta=" + std::to_string(beta));
    require(association->weight > 0.99F && association->activation > 0.99F,
            "coalesced transition update did not preserve bounded learning");

    HierarchicalPredictionNetwork sequential_network(PredictionNetworkConfig{
        .total_neuron_count = 4000U,
        .neurons_per_layer = 1000U,
        .additional_layer_count = 0U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 512U,
    });
    for (const HierarchicalTokenTrace& trace : traces) {
        sequential_network.observe(trace);
    }
    const auto sequential_association =
        sequential_network.inspect_association(PredictionLayer::word, alpha, beta);
    require(sequential_association.has_value() &&
                std::fabs(sequential_association->weight - association->weight) < 1.0e-5F &&
                std::fabs(sequential_association->activation - association->activation) <
                    1.0e-5F,
            "coalesced learning diverged from ordered online recurrence");
    require(network.state().step == 1U,
            "worker-local learning delta batch performed multiple route passes");
}

void teacher_feedback_contract() {
    using namespace agentari::text::hierarchical;

    HierarchicalTokenizer tokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 64U,
            .minimum_part_length = 2U,
            .maximum_part_length = 8U,
            .minimum_distinct_words = 2U,
        },
        .vocabulary_capacity = 128U,
    });
    HierarchicalPredictionNetwork network(PredictionNetworkConfig{
        .total_neuron_count = 4000U,
        .neurons_per_layer = 1000U,
        .additional_layer_count = 0U,
        .min_neurons_per_layer = 16U,
        .maximum_transition_neurons = 512U,
    });
    TeacherFeedbackTrainer trainer(tokenizer, network);

    (void)trainer.observe("unhappy unkind");
    const PartId un = tokenizer.parts().lookup("un");
    require(un != 0U, "teacher test did not learn the shared word part");
    const HierarchicalTokenTrace input_trace = trainer.observe("unhappy unkind");
    const TeacherFeedback positive{
        .segmentation_quality = 1.0F,
        .semantic_quality = 1.0F,
        .grammar_quality = 1.0F,
        .context_relevance = 1.0F,
        .confidence = 1.0F,
        .corrected_text = "unhappy unkind",
    };
    const TeacherFeedbackApplication application =
        trainer.apply(input_trace, "unhappy unkind", positive);
    require(application.applied && application.learned_candidate &&
                application.learned_correction &&
                application.candidate_tokens > 0U,
            "teacher feedback was not applied or trusted correction was not learned");
    const LearnedPart* supported = tokenizer.parts().part(un);
    require(supported != nullptr && supported->teacher_support > 0.5F,
            "positive teacher feedback did not support the learned part");

    TeacherFeedback negative = positive;
    negative.segmentation_quality = 0.0F;
    negative.semantic_quality = 0.0F;
    negative.grammar_quality = 0.0F;
    negative.context_relevance = 0.0F;
    negative.confidence = 1.0F;
    negative.corrected_text.clear();
    (void)trainer.apply(input_trace, "unhappy unkind", negative);
    supported = tokenizer.parts().part(un);
    require(supported != nullptr && supported->teacher_support < 0.76F,
            "negative teacher feedback did not remain bounded and visible");
    require(trainer.state().presented == 2U && trainer.state().applied == 2U &&
                trainer.state().learned_candidates == 1U &&
                trainer.state().learned_corrections == 1U,
            "teacher feedback accounting is inconsistent");
}

}  // namespace

int main() {
    try {
        predictive_layers_learn();
        batched_observation_contract();
        worker_local_delta_contract();
        teacher_feedback_contract();
        std::cout << "hierarchical prediction tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "hierarchical prediction tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
