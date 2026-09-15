#include "agentari/SemanticPartLearner.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace agentari::text::hierarchical {
namespace {

bool better_part(const LearnedPart& left, const LearnedPart& right) {
    if (left.compositionality_score != right.compositionality_score) {
        return left.compositionality_score > right.compositionality_score;
    }
    if (left.reuse_score != right.reuse_score) {
        return left.reuse_score > right.reuse_score;
    }
    if (left.distinct_word_count != right.distinct_word_count) {
        return left.distinct_word_count > right.distinct_word_count;
    }
    if (left.text.size() != right.text.size()) {
        return left.text.size() > right.text.size();
    }
    return left.text < right.text;
}

}  // namespace

SemanticPartLearner::SemanticPartLearner(SemanticPartLearnerConfig config)
    : config_(config) {
    if (config_.capacity == 0U || config_.minimum_part_length < 2U ||
        config_.maximum_part_length < config_.minimum_part_length ||
        config_.minimum_distinct_words < 2U) {
        throw std::invalid_argument("invalid semantic part learner configuration");
    }
    parts_.reserve(config_.capacity + 1U);
    parts_.push_back(LearnedPart{});  // ID zero is reserved for character fallback.
}

std::string SemanticPartLearner::normalize(const std::string_view word) {
    std::string result;
    result.reserve(word.size());
    for (const unsigned char value : word) {
        result.push_back(static_cast<char>(std::tolower(value)));
    }
    return result;
}

void SemanticPartLearner::observe_word(
    const std::string_view word,
    const std::span<const std::string_view> context_words) {
    observe_word_stats(word, context_words);
    promote_candidates();
    refresh_promoted_parts();
}

void SemanticPartLearner::observe_word_stats(
    const std::string_view word,
    const std::span<const std::string_view> context_words) {
    const std::string normalized = normalize(word);
    if (normalized.size() < config_.minimum_part_length) {
        return;
    }

    normalized_context_scratch_.clear();
    normalized_context_scratch_.reserve(context_words.size());
    for (const std::string_view context_word : context_words) {
        std::string context = normalize(context_word);
        if (!context.empty() && context != normalized) {
            normalized_context_scratch_.push_back(std::move(context));
        }
    }

    observed_candidate_scratch_.clear();
    for (std::size_t begin = 0U; begin < normalized.size(); ++begin) {
        const std::size_t maximum_length = std::min(
            config_.maximum_part_length, normalized.size() - begin);
        for (std::size_t length = config_.minimum_part_length; length <= maximum_length; ++length) {
            if (begin == 0U && length == normalized.size()) {
                continue;  // The complete word belongs to the vocabulary layer.
            }
            observed_candidate_scratch_.insert(normalized.substr(begin, length));
        }
    }

    for (const std::string& candidate : observed_candidate_scratch_) {
        CandidateStats& stats = candidates_[candidate];
        ++stats.observations;
        stats.words.insert(normalized);
        for (const std::string& context : normalized_context_scratch_) {
            ++stats.context_observations;
            stats.context_carriers[context].insert(normalized);
        }
    }
}

void SemanticPartLearner::observe_text(const std::string_view input) {
    const AtomicTokenizer atomic;
    const std::vector<WordSpan> spans = atomic.word_spans(input);
    normalized_word_scratch_.clear();
    normalized_word_scratch_.reserve(spans.size());
    for (const WordSpan& span : spans) {
        normalized_word_scratch_.push_back(normalize(span.text));
    }
    for (std::size_t index = 0U; index < normalized_word_scratch_.size(); ++index) {
        const std::size_t begin = index > config_.context_radius
                                       ? index - config_.context_radius
                                       : 0U;
        const std::size_t end = std::min(normalized_word_scratch_.size(),
                                         index + config_.context_radius + 1U);
        context_view_scratch_.clear();
        context_view_scratch_.reserve(end - begin);
        for (std::size_t neighbor = begin; neighbor < end; ++neighbor) {
            if (neighbor != index) {
                context_view_scratch_.push_back(normalized_word_scratch_[neighbor]);
            }
        }
        observe_word_stats(normalized_word_scratch_[index],
                           std::span<const std::string_view>(context_view_scratch_.data(),
                                                              context_view_scratch_.size()));
    }
    // Candidate promotion is intentionally amortized over the complete input
    // record. Promoting after every word repeatedly rescanned the full
    // candidate table and made long documents disproportionately expensive.
    promote_candidates();
    refresh_promoted_parts();
}

void SemanticPartLearner::apply_teacher_feedback(const PartAnalysis& analysis,
                                                 const float quality) {
    if (!std::isfinite(quality)) {
        return;
    }
    const float signal = 2.0F * std::clamp(quality, 0.0F, 1.0F) - 1.0F;
    for (const PartSpan& span : analysis.spans) {
        if (!span.learned || span.id == 0U || span.id >= parts_.size()) {
            continue;
        }
        const std::string& text = parts_[span.id].text;
        const auto candidate = candidates_.find(text);
        if (candidate == candidates_.end()) {
            continue;
        }
        ++candidate->second.teacher_observations;
        candidate->second.teacher_score += signal;
    }
    // Teacher feedback changes only the parts represented by this analysis.
    // Refreshing the complete promoted bank here made every candidate answer
    // rescan all learned parts, even when only a few spans were touched.
    for (const PartSpan& span : analysis.spans) {
        if (span.learned) {
            refresh_promoted_part(span.id);
        }
    }
}

float SemanticPartLearner::teacher_support(const CandidateStats& stats) noexcept {
    if (stats.teacher_observations == 0U) {
        return 0.5F;
    }
    const float average = stats.teacher_score /
                          static_cast<float>(stats.teacher_observations);
    return std::clamp(0.5F + 0.5F * average, 0.0F, 1.0F);
}

void SemanticPartLearner::promote_candidates() {
    if (parts_.size() - 1U >= config_.capacity) {
        return;
    }

    eligible_scratch_.clear();
    eligible_scratch_.reserve(candidates_.size());
    for (const auto& [text, stats] : candidates_) {
        if (stats.words.size() < config_.minimum_distinct_words || ids_.contains(text)) {
            continue;
        }
        const float reuse_score = static_cast<float>(stats.words.size()) *
                                  static_cast<float>(text.size());
        std::size_t context_carrier_total = 0U;
        for (const auto& [context, carriers] : stats.context_carriers) {
            (void)context;
            context_carrier_total += carriers.size();
        }
        const float contextual_support = stats.context_carriers.empty() || stats.words.empty()
                                            ? 0.0F
                                            : static_cast<float>(context_carrier_total) /
                                                  static_cast<float>(stats.context_carriers.size() *
                                                                     stats.words.size());
        const float word_reuse = std::min(1.0F,
                                          static_cast<float>(stats.words.size()) / 4.0F);
        const float base_score = 0.5F * word_reuse + 0.5F * contextual_support;
        const float supported_score = teacher_support(stats);
        const float compositionality_score = stats.teacher_observations == 0U
                                                 ? base_score
                                                 : 0.8F * base_score + 0.2F * supported_score;
        eligible_scratch_.push_back(LearnedPart{
            .id = 0U,
            .text = text,
            .observations = stats.observations,
            .distinct_word_count = stats.words.size(),
            .context_observations = stats.context_observations,
            .distinct_context_word_count = stats.context_carriers.size(),
            .reuse_score = reuse_score,
            .contextual_support = contextual_support,
            .compositionality_score = compositionality_score,
            .teacher_support = supported_score,
        });
    }
    std::sort(eligible_scratch_.begin(), eligible_scratch_.end(), better_part);

    for (LearnedPart& candidate : eligible_scratch_) {
        if (parts_.size() - 1U >= config_.capacity) {
            break;
        }
        if (ids_.contains(candidate.text)) {
            continue;
        }
        candidate.id = static_cast<PartId>(parts_.size());
        ids_.emplace(candidate.text, candidate.id);
        parts_.push_back(std::move(candidate));
        const unsigned char first_byte = static_cast<unsigned char>(parts_.back().text.front());
        parts_by_first_byte_[first_byte].push_back(parts_.back().id);
    }
}

void SemanticPartLearner::refresh_promoted_parts() {
    for (std::size_t index = 1U; index < parts_.size(); ++index) {
        refresh_promoted_part(static_cast<PartId>(index));
    }
}

void SemanticPartLearner::refresh_promoted_part(const PartId id) {
    if (id == 0U || id >= parts_.size()) {
        return;
    }
    LearnedPart& part = parts_[id];
    const auto candidate = candidates_.find(part.text);
    if (candidate == candidates_.end()) {
        return;
    }
    const CandidateStats& stats = candidate->second;
    std::size_t context_carrier_total = 0U;
    for (const auto& [context, carriers] : stats.context_carriers) {
        (void)context;
        context_carrier_total += carriers.size();
    }
    const float contextual_support = stats.context_carriers.empty() || stats.words.empty()
                                        ? 0.0F
                                        : static_cast<float>(context_carrier_total) /
                                              static_cast<float>(stats.context_carriers.size() *
                                                                 stats.words.size());
    const float word_reuse = std::min(1.0F,
                                      static_cast<float>(stats.words.size()) / 4.0F);
    const float base_score = 0.5F * word_reuse + 0.5F * contextual_support;
    const float supported_score = teacher_support(stats);
    part.observations = stats.observations;
    part.distinct_word_count = stats.words.size();
    part.context_observations = stats.context_observations;
    part.distinct_context_word_count = stats.context_carriers.size();
    part.reuse_score = static_cast<float>(stats.words.size()) *
                       static_cast<float>(part.text.size());
    part.contextual_support = contextual_support;
    part.teacher_support = supported_score;
    part.compositionality_score = stats.teacher_observations == 0U
                                      ? base_score
                                      : 0.8F * base_score + 0.2F * supported_score;
}

PartAnalysis SemanticPartLearner::analyze(const std::string_view word) const {
    PartAnalysis result;
    result.source = normalize(word);
    const std::size_t length = result.source.size();
    if (length == 0U) {
        return result;
    }

    struct Choice {
        float score{0.0F};
        std::size_t next{0U};
        PartId id{0U};
        bool learned{false};
    };
    std::vector<Choice> choices(length + 1U);
    for (std::size_t position = length; position-- > 0U;) {
        choices[position] = Choice{
            .score = choices[position + 1U].score,
            .next = position + 1U,
            .id = 0U,
            .learned = false,
        };
        const auto& candidates = parts_by_first_byte_[
            static_cast<unsigned char>(result.source[position])];
        for (const PartId candidate_id : candidates) {
            const LearnedPart& candidate = parts_[candidate_id];
            if (position + candidate.text.size() > length ||
                result.source.compare(position, candidate.text.size(), candidate.text) != 0) {
                continue;
            }
            const std::size_t next = position + candidate.text.size();
            const float score = choices[next].score + candidate.reuse_score;
            if (score > choices[position].score ||
                (score == choices[position].score && next > choices[position].next)) {
                choices[position] = Choice{
                    .score = score,
                    .next = next,
                    .id = candidate.id,
                    .learned = true,
                };
            }
        }
    }

    std::size_t position = 0U;
    std::size_t learned_bytes = 0U;
    while (position < length) {
        const Choice& choice = choices[position];
        result.spans.push_back(PartSpan{
            .id = choice.id,
            .begin = position,
            .end = choice.next,
            .learned = choice.learned,
        });
        if (choice.learned) {
            learned_bytes += choice.next - position;
        }
        position = choice.next;
    }
    result.learned_coverage = static_cast<float>(learned_bytes) / static_cast<float>(length);
    return result;
}

PartId SemanticPartLearner::lookup(const std::string_view part) const noexcept {
    const auto found = ids_.find(normalize(part));
    return found == ids_.end() ? 0U : found->second;
}

const LearnedPart* SemanticPartLearner::part(const PartId id) const noexcept {
    return id < parts_.size() && id != 0U ? &parts_[id] : nullptr;
}

const std::vector<LearnedPart>& SemanticPartLearner::learned_parts() const noexcept {
    return parts_;
}

const SemanticPartLearnerConfig& SemanticPartLearner::config() const noexcept {
    return config_;
}

std::size_t SemanticPartLearner::size() const noexcept {
    return parts_.size() - 1U;
}

}  // namespace agentari::text::hierarchical
