#pragma once

#include "agentari/HierarchicalPrediction.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace agentari::text::hierarchical {

// All values are normalized to [0, 1]. A judge is advisory: the trainer
// clamps malformed or overconfident records before they can affect state.
struct TeacherFeedback {
    float segmentation_quality{0.5F};
    float semantic_quality{0.5F};
    float grammar_quality{0.5F};
    float context_relevance{0.5F};
    float confidence{0.0F};
    std::string corrected_text;
};

struct TeacherFeedbackConfig {
    // The first judge pilot was heavily overconfident. These stricter defaults
    // reject mediocre or unrelated responses before they become positive data.
    float accepted_candidate_threshold{0.86F};
    float accepted_correction_threshold{0.86F};
    float minimum_candidate_confidence{0.90F};
    float minimum_correction_confidence{0.90F};
    bool learn_accepted_candidates{true};
    bool learn_teacher_corrections{true};
};

struct TeacherFeedbackApplication {
    bool applied{false};
    bool learned_candidate{false};
    bool learned_correction{false};
    std::size_t candidate_tokens{0U};
    float overall_quality{0.0F};
};

struct TeacherFeedbackState {
    std::uint64_t presented{0U};
    std::uint64_t applied{0U};
    std::uint64_t learned_candidates{0U};
    std::uint64_t learned_corrections{0U};
};

// Coordinates tokenizer and predictor updates while leaving the teacher
// model outside the process. A Python/llama.cpp judge can emit the feedback
// schema and this class applies it deterministically.
class TeacherFeedbackTrainer {
public:
    TeacherFeedbackTrainer(HierarchicalTokenizer& tokenizer,
                           HierarchicalPredictionNetwork& network,
                           TeacherFeedbackConfig config = {});

    [[nodiscard]] HierarchicalTokenTrace observe(std::string_view input);
    [[nodiscard]] TeacherFeedbackApplication apply(
        const HierarchicalTokenTrace& input_trace,
        std::string_view candidate,
        const TeacherFeedback& feedback);

    [[nodiscard]] const TeacherFeedbackConfig& config() const noexcept;
    [[nodiscard]] const TeacherFeedbackState& state() const noexcept;

private:
    HierarchicalTokenizer* tokenizer_;
    HierarchicalPredictionNetwork* network_;
    TeacherFeedbackConfig config_;
    TeacherFeedbackState state_;
};

}  // namespace agentari::text::hierarchical
