#pragma once

#include "agentari/WordVocabulary.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace agentari::text::hierarchical {

struct ContextTokenizerConfig {
    std::size_t recent_word_limit{256U};
};

struct ContextSnapshot {
    std::vector<WordId> recent_words;
    std::uint64_t observed_words{0U};
};

struct ContextPrediction {
    WordId word{WordVocabulary::unknown_id};
    std::uint64_t count{0U};
    float probability{0.0F};
};

class ContextTokenizer {
public:
    explicit ContextTokenizer(ContextTokenizerConfig config = {});

    void observe(std::span<const WordId> words);
    [[nodiscard]] ContextSnapshot snapshot() const;
    [[nodiscard]] std::vector<ContextPrediction> predict_next(std::size_t top_k = 5U) const;
    [[nodiscard]] const ContextTokenizerConfig& config() const noexcept;

private:
    ContextTokenizerConfig config_;
    std::vector<WordId> recent_words_;
    std::uint64_t observed_words_{0U};
    WordId previous_word_{WordVocabulary::unknown_id};
    std::unordered_map<WordId, std::unordered_map<WordId, std::uint64_t>> transitions_;
};

}  // namespace agentari::text::hierarchical
