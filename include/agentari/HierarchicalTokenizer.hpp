#pragma once

#include "agentari/AtomicTokenizer.hpp"
#include "agentari/ContextTokenizer.hpp"
#include "agentari/SemanticPartLearner.hpp"
#include "agentari/WordVocabulary.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace agentari::text::hierarchical {

struct HierarchicalTokenizerConfig {
    SemanticPartLearnerConfig parts{};
    std::size_t vocabulary_capacity{100000U};
    ContextTokenizerConfig context{};
};

struct WordTrace {
    WordSpan span;
    WordId word{WordVocabulary::unknown_id};
    PartAnalysis parts;
};

struct HierarchicalTokenTrace {
    std::string source;
    std::vector<AtomicToken> atomic_tokens;
    std::vector<WordTrace> words;
    ContextSnapshot context;
};

class HierarchicalTokenizer {
public:
    explicit HierarchicalTokenizer(HierarchicalTokenizerConfig config = {});

    // encode() is read-only. observe() learns parts and words, then advances
    // the context layer with the resulting whole-word IDs.
    [[nodiscard]] HierarchicalTokenTrace encode(std::string_view input) const;
    [[nodiscard]] HierarchicalTokenTrace observe(std::string_view input);
    void apply_teacher_feedback(const HierarchicalTokenTrace& trace, float quality);

    [[nodiscard]] const AtomicTokenizer& atomic() const noexcept;
    [[nodiscard]] const SemanticPartLearner& parts() const noexcept;
    [[nodiscard]] const WordVocabulary& words() const noexcept;
    [[nodiscard]] const ContextTokenizer& context() const noexcept;

private:
    AtomicTokenizer atomic_;
    SemanticPartLearner parts_;
    WordVocabulary words_;
    ContextTokenizer context_;
    // Ordered observe() reuses this bridge into the context tokenizer. Trace
    // analysis remains read-only and may run on the persistent worker pool.
    std::vector<WordId> word_id_scratch_;

    [[nodiscard]] HierarchicalTokenTrace trace(std::string_view input) const;
};

}  // namespace agentari::text::hierarchical
