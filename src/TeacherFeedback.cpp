#include "agentari/TeacherFeedback.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace agentari::text::hierarchical {
namespace {

float safe_quality(const float value) {
    return std::isfinite(value) ? std::clamp(value, 0.0F, 1.0F) : 0.5F;
}

}  // namespace

TeacherFeedbackTrainer::TeacherFeedbackTrainer(
    HierarchicalTokenizer& tokenizer,
    HierarchicalPredictionNetwork& network,
    TeacherFeedbackConfig config)
    : tokenizer_(&tokenizer), network_(&network), config_(config) {
    if (!std::isfinite(config_.accepted_candidate_threshold) ||
        !std::isfinite(config_.accepted_correction_threshold) ||
        !std::isfinite(config_.minimum_candidate_confidence) ||
        !std::isfinite(config_.minimum_correction_confidence) ||
        config_.accepted_candidate_threshold < 0.0F ||
        config_.accepted_candidate_threshold > 1.0F ||
        config_.accepted_correction_threshold < 0.0F ||
        config_.accepted_correction_threshold > 1.0F ||
        config_.minimum_candidate_confidence < 0.0F ||
        config_.minimum_candidate_confidence > 1.0F ||
        config_.minimum_correction_confidence < 0.0F ||
        config_.minimum_correction_confidence > 1.0F) {
        throw std::invalid_argument("invalid teacher feedback configuration");
    }
}

HierarchicalTokenTrace TeacherFeedbackTrainer::observe(const std::string_view input) {
    ++state_.presented;
    return network_->train_step(*tokenizer_, input);
}

TeacherFeedbackApplication TeacherFeedbackTrainer::apply(
    const HierarchicalTokenTrace& input_trace,
    const std::string_view candidate,
    const TeacherFeedback& feedback) {
    const float segmentation = safe_quality(feedback.segmentation_quality);
    const float semantic = safe_quality(feedback.semantic_quality);
    const float grammar = safe_quality(feedback.grammar_quality);
    const float context = safe_quality(feedback.context_relevance);
    const float confidence = safe_quality(feedback.confidence);
    const float overall = std::clamp(0.35F * semantic + 0.25F * grammar +
                                         0.25F * context + 0.15F * segmentation,
                                     0.0F, 1.0F);

    // Context quality is feedback about the input-to-prediction relationship;
    // it adjusts existing prompt associations. Candidate quality updates the
    // candidate's own segmentation and prediction paths.
    network_->apply_teacher_feedback(input_trace, context, confidence);

    TeacherFeedbackApplication result{
        .applied = true,
        .learned_candidate = false,
        .learned_correction = false,
        .candidate_tokens = 0U,
        .overall_quality = overall,
    };
    if (!candidate.empty()) {
        const HierarchicalTokenTrace candidate_trace = tokenizer_->encode(candidate);
        result.candidate_tokens = candidate_trace.atomic_tokens.size();
        tokenizer_->apply_teacher_feedback(candidate_trace, segmentation);
        network_->apply_teacher_feedback(candidate_trace, overall, confidence);

        // A high-quality candidate is a positive online example. Train it
        // only after the frozen judge crosses both gates; low-quality output
        // can still receive a negative bounded association signal above.
        if (config_.learn_accepted_candidates &&
            overall >= config_.accepted_candidate_threshold &&
            confidence >= config_.minimum_candidate_confidence) {
            const HierarchicalTokenTrace learned_candidate =
                network_->train_step(*tokenizer_, candidate);
            // Apply the signal again after observation so transitions and
            // parts that were newly created by this accepted example receive
            // the same positive teacher trace.
            tokenizer_->apply_teacher_feedback(learned_candidate, segmentation);
            network_->apply_teacher_feedback(learned_candidate, overall, confidence);
            result.learned_candidate = true;
            ++state_.learned_candidates;
        }
    }

    if (config_.learn_teacher_corrections && !feedback.corrected_text.empty() &&
        overall >= config_.accepted_correction_threshold &&
        confidence >= config_.minimum_correction_confidence) {
        const HierarchicalTokenTrace learned_correction =
            network_->train_step(*tokenizer_, feedback.corrected_text);
        tokenizer_->apply_teacher_feedback(learned_correction, segmentation);
        network_->apply_teacher_feedback(learned_correction, overall, confidence);
        result.learned_correction = true;
        ++state_.learned_corrections;
    }
    ++state_.applied;
    return result;
}

const TeacherFeedbackConfig& TeacherFeedbackTrainer::config() const noexcept {
    return config_;
}

const TeacherFeedbackState& TeacherFeedbackTrainer::state() const noexcept {
    return state_;
}

}  // namespace agentari::text::hierarchical
