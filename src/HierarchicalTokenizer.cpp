#include "agentari/HierarchicalTokenizer.hpp"
#include "agentari/Parallel.hpp"

#include <span>

namespace agentari::text::hierarchical {

HierarchicalTokenizer::HierarchicalTokenizer(HierarchicalTokenizerConfig config)
    : parts_(config.parts), words_(config.vocabulary_capacity), context_(config.context) {
    word_id_scratch_.reserve(config.context.recent_word_limit);
}

HierarchicalTokenTrace HierarchicalTokenizer::trace(const std::string_view input) const {
    HierarchicalTokenTrace result;
    result.source = std::string(input);
    result.atomic_tokens = atomic_.encode(input);
    const std::vector<WordSpan> spans = atomic_.word_spans(input);
    result.words.resize(spans.size());

    // This section is intentionally read-only: observe() performs all
    // vocabulary, semantic-part, and context mutations before calling trace,
    // while encode() is const. Each word can therefore be analyzed on a
    // separate worker without changing the order-sensitive online learner.
    parallel::Config parallel_config;
    parallel_config.grain_size = 8U;
    parallel_config.minimum_parallel_work = 128U;
    parallel::parallel_for(0U, spans.size(), [&](const std::size_t index) {
        const WordSpan& span = spans[index];
        result.words[index] = WordTrace{
            .span = span,
            .word = words_.lookup(span.text),
            .parts = parts_.analyze(span.text),
        };
    }, parallel_config);
    result.context = context_.snapshot();
    return result;
}

HierarchicalTokenTrace HierarchicalTokenizer::encode(const std::string_view input) const {
    return trace(input);
}

HierarchicalTokenTrace HierarchicalTokenizer::observe(const std::string_view input) {
    const std::vector<WordSpan> spans = atomic_.word_spans(input);
    parts_.observe_text(input);
    for (const WordSpan& span : spans) {
        (void)words_.observe_word(span.text);
    }

    HierarchicalTokenTrace result = trace(input);
    word_id_scratch_.clear();
    word_id_scratch_.reserve(result.words.size());
    for (const WordTrace& word : result.words) {
        word_id_scratch_.push_back(word.word);
    }
    context_.observe(std::span<const WordId>(word_id_scratch_.data(),
                                             word_id_scratch_.size()));
    result.context = context_.snapshot();
    return result;
}

void HierarchicalTokenizer::apply_teacher_feedback(
    const HierarchicalTokenTrace& trace,
    const float quality) {
    for (const WordTrace& word : trace.words) {
        parts_.apply_teacher_feedback(word.parts, quality);
    }
}

const AtomicTokenizer& HierarchicalTokenizer::atomic() const noexcept {
    return atomic_;
}

const SemanticPartLearner& HierarchicalTokenizer::parts() const noexcept {
    return parts_;
}

const WordVocabulary& HierarchicalTokenizer::words() const noexcept {
    return words_;
}

const ContextTokenizer& HierarchicalTokenizer::context() const noexcept {
    return context_;
}

}  // namespace agentari::text::hierarchical
