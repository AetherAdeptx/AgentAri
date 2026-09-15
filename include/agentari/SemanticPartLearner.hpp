#pragma once

#include "agentari/AtomicTokenizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace agentari::text::hierarchical {

using PartId = std::uint32_t;

struct SemanticPartLearnerConfig {
    std::size_t capacity{1024U};
    std::size_t minimum_part_length{2U};
    std::size_t maximum_part_length{16U};
    std::size_t minimum_distinct_words{2U};
    std::size_t context_radius{2U};
};

struct LearnedPart {
    PartId id{0U};
    std::string text;
    std::uint64_t observations{0U};
    std::size_t distinct_word_count{0U};
    std::uint64_t context_observations{0U};
    std::size_t distinct_context_word_count{0U};
    float reuse_score{0.0F};
    float contextual_support{0.0F};
    float compositionality_score{0.0F};
    float teacher_support{0.5F};
};

struct PartSpan {
    PartId id{0U};
    std::size_t begin{0U};
    std::size_t end{0U};
    bool learned{false};
};

struct PartAnalysis {
    std::string source;
    std::vector<PartSpan> spans;
    float learned_coverage{0.0F};
};

class SemanticPartLearner {
public:
    explicit SemanticPartLearner(SemanticPartLearnerConfig config = {});

    // The learner starts with no word parts. It discovers repeated contiguous
    // spans from observed words; no morphology table or precomputed pieces are
    // loaded by this class.
    void observe_word(std::string_view word,
                      std::span<const std::string_view> context_words = {});
    void observe_text(std::string_view input);
    // Applies a bounded external-teacher signal to the learned spans already
    // present in an analysis. The teacher cannot create arbitrary parts or
    // bypass the learner's capacity and compositionality rules.
    void apply_teacher_feedback(const PartAnalysis& analysis, float quality);

    [[nodiscard]] PartAnalysis analyze(std::string_view word) const;
    [[nodiscard]] PartId lookup(std::string_view part) const noexcept;
    [[nodiscard]] const LearnedPart* part(PartId id) const noexcept;
    [[nodiscard]] const std::vector<LearnedPart>& learned_parts() const noexcept;
    [[nodiscard]] const SemanticPartLearnerConfig& config() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct CandidateStats {
        std::uint64_t observations{0U};
        std::unordered_set<std::string> words;
        std::uint64_t context_observations{0U};
        std::unordered_map<std::string, std::unordered_set<std::string>> context_carriers;
        std::uint64_t teacher_observations{0U};
        float teacher_score{0.0F};
    };

    SemanticPartLearnerConfig config_;
    std::unordered_map<std::string, CandidateStats> candidates_;
    std::unordered_map<std::string, PartId> ids_;
    std::vector<LearnedPart> parts_;
    // Read-only analysis is indexed by the first byte of a normalized word,
    // avoiding a full scan of every promoted part at every character offset.
    std::array<std::vector<PartId>, 256U> parts_by_first_byte_;
    // These buffers belong to the ordered mutating learner and retain capacity
    // between records. They are not shared with parallel trace analysis.
    std::vector<std::string> normalized_word_scratch_;
    std::vector<std::string> normalized_context_scratch_;
    std::vector<std::string_view> context_view_scratch_;
    std::unordered_set<std::string> observed_candidate_scratch_;
    std::vector<LearnedPart> eligible_scratch_;

    [[nodiscard]] static std::string normalize(std::string_view word);
    void observe_word_stats(std::string_view word,
                            std::span<const std::string_view> context_words);
    [[nodiscard]] static float teacher_support(const CandidateStats& stats) noexcept;
    void promote_candidates();
    void refresh_promoted_parts();
    void refresh_promoted_part(PartId id);
};

}  // namespace agentari::text::hierarchical
