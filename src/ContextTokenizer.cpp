#include "agentari/ContextTokenizer.hpp"

#include <algorithm>
#include <stdexcept>

namespace agentari::text::hierarchical {

ContextTokenizer::ContextTokenizer(ContextTokenizerConfig config)
    : config_(config) {
    if (config_.recent_word_limit == 0U) {
        throw std::invalid_argument("context recent-word limit must be non-zero");
    }
    recent_words_.reserve(config_.recent_word_limit);
}

void ContextTokenizer::observe(const std::span<const WordId> words) {
    for (const WordId word : words) {
        if (word == WordVocabulary::unknown_id) {
            continue;
        }
        if (previous_word_ != WordVocabulary::unknown_id) {
            ++transitions_[previous_word_][word];
        }
        previous_word_ = word;
        recent_words_.push_back(word);
        if (recent_words_.size() > config_.recent_word_limit) {
            recent_words_.erase(recent_words_.begin());
        }
        ++observed_words_;
    }
}

ContextSnapshot ContextTokenizer::snapshot() const {
    return ContextSnapshot{
        .recent_words = recent_words_,
        .observed_words = observed_words_,
    };
}

std::vector<ContextPrediction> ContextTokenizer::predict_next(const std::size_t top_k) const {
    if (top_k == 0U || recent_words_.empty()) {
        return {};
    }
    const auto transition = transitions_.find(recent_words_.back());
    if (transition == transitions_.end()) {
        return {};
    }

    std::vector<ContextPrediction> result;
    result.reserve(transition->second.size());
    std::uint64_t total = 0U;
    for (const auto& [word, count] : transition->second) {
        result.push_back(ContextPrediction{.word = word, .count = count});
        total += count;
    }
    std::sort(result.begin(), result.end(), [](const ContextPrediction& left,
                                               const ContextPrediction& right) {
        if (left.count != right.count) {
            return left.count > right.count;
        }
        return left.word < right.word;
    });
    if (result.size() > top_k) {
        result.resize(top_k);
    }
    for (ContextPrediction& prediction : result) {
        prediction.probability = total == 0U
                                     ? 0.0F
                                     : static_cast<float>(prediction.count) /
                                           static_cast<float>(total);
    }
    return result;
}

const ContextTokenizerConfig& ContextTokenizer::config() const noexcept {
    return config_;
}

}  // namespace agentari::text::hierarchical
