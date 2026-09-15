#include "agentari/HierarchicalTokenizer.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void hierarchical_tokenizer_contract() {
    using namespace agentari::text::hierarchical;

    HierarchicalTokenizer tokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 32U,
            .minimum_part_length = 2U,
            .maximum_part_length = 8U,
            .minimum_distinct_words = 2U,
        },
        .vocabulary_capacity = 64U,
        .context = ContextTokenizerConfig{.recent_word_limit = 8U},
    });

    const std::string sample = "unhappy happiness. 2026!";
    const auto atomic = tokenizer.atomic().encode(sample);
    require(tokenizer.atomic().decode(atomic) == sample,
            "atomic layer did not reconstruct the input");
    require(atomic[0U].kind == AtomicKind::character, "letters were not atomic characters");
    require(atomic.back().kind == AtomicKind::punctuation,
            "punctuation was not preserved as an atomic token");

    (void)tokenizer.observe("unhappy happiness");
    (void)tokenizer.observe("unkind darkness");
    (void)tokenizer.observe("happiness is good; darkness is present");
    const auto trace = tokenizer.observe(sample);
    require(trace.words.size() == 2U, "word layer did not find both words");
    require(trace.words[0U].word != WordVocabulary::unknown_id,
            "observed whole word was not added to the vocabulary");
    require(tokenizer.words().lookup("happiness") == trace.words[1U].word,
            "word lookup disagrees with the hierarchical trace");
    require(tokenizer.parts().lookup("un") != 0U,
            "repeated prefix was not discovered as a learned part");
    require(tokenizer.parts().lookup("ness") != 0U,
            "repeated suffix was not discovered as a learned part");
    const LearnedPart* ness = tokenizer.parts().part(tokenizer.parts().lookup("ness"));
    require(ness != nullptr && ness->context_observations > 0U,
            "learned part did not retain neighboring-word evidence");
    require(ness->distinct_context_word_count > 0U && ness->compositionality_score > 0.0F,
            "learned part did not receive a compositionality score");

    const PartAnalysis analysis = tokenizer.parts().analyze("unhappiness");
    require(analysis.learned_coverage > 0.0F,
            "learned part analysis did not cover any word characters");
    require(!analysis.spans.empty() && analysis.spans.back().end == analysis.source.size(),
            "part analysis did not cover the complete word");

    const auto context_predictions = tokenizer.context().predict_next(3U);
    require(!context_predictions.empty(), "context layer did not learn transitions");
    require(trace.context.recent_words.size() <= 8U,
            "context layer exceeded its configured recent-word limit");
    require(trace.context.observed_words > 0U, "context layer did not observe words");
}

}  // namespace

int main() {
    try {
        hierarchical_tokenizer_contract();
        std::cout << "hierarchical tokenizer tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "hierarchical tokenizer tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
