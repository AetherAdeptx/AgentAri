#include "agentari/WordVocabulary.hpp"

#include <cctype>
#include <stdexcept>

namespace agentari::text::hierarchical {

WordVocabulary::WordVocabulary(const std::size_t capacity)
    : capacity_(capacity) {
    if (capacity_ == 0U) {
        throw std::invalid_argument("word vocabulary capacity must be non-zero");
    }
    entries_.reserve(capacity_ + 1U);
    entries_.push_back(WordEntry{});  // ID zero is the unknown word.
}

std::string WordVocabulary::normalize(const std::string_view word) {
    std::string result;
    result.reserve(word.size());
    for (const unsigned char value : word) {
        result.push_back(static_cast<char>(std::tolower(value)));
    }
    return result;
}

WordId WordVocabulary::observe_word(const std::string_view word,
                                     const std::size_t frequency_rank) {
    const std::string normalized = normalize(word);
    if (normalized.empty()) {
        return unknown_id;
    }
    const auto found = ids_.find(normalized);
    if (found != ids_.end()) {
        WordEntry& entry = entries_[found->second];
        ++entry.observations;
        if (frequency_rank != 0U) {
            entry.frequency_rank = frequency_rank;
        }
        return found->second;
    }
    if (entries_.size() - 1U >= capacity_) {
        return unknown_id;
    }

    const WordId id = static_cast<WordId>(entries_.size());
    ids_.emplace(normalized, id);
    entries_.push_back(WordEntry{
        .id = id,
        .text = normalized,
        .frequency_rank = frequency_rank,
        .observations = 1U,
    });
    return id;
}

void WordVocabulary::observe_text(const std::string_view input) {
    const AtomicTokenizer atomic;
    for (const WordSpan& span : atomic.word_spans(input)) {
        (void)observe_word(span.text);
    }
}

WordId WordVocabulary::lookup(const std::string_view word) const noexcept {
    const auto found = ids_.find(normalize(word));
    return found == ids_.end() ? unknown_id : found->second;
}

std::string_view WordVocabulary::text(const WordId id) const noexcept {
    const WordEntry* value = entry(id);
    return value == nullptr ? std::string_view{} : value->text;
}

const WordEntry* WordVocabulary::entry(const WordId id) const noexcept {
    return id < entries_.size() && id != unknown_id ? &entries_[id] : nullptr;
}

std::vector<WordId> WordVocabulary::encode_text(const std::string_view input) const {
    const AtomicTokenizer atomic;
    std::vector<WordId> result;
    for (const WordSpan& span : atomic.word_spans(input)) {
        result.push_back(lookup(span.text));
    }
    return result;
}

std::size_t WordVocabulary::size() const noexcept {
    return entries_.size() - 1U;
}

std::size_t WordVocabulary::capacity() const noexcept {
    return capacity_;
}

const std::vector<WordEntry>& WordVocabulary::entries() const noexcept {
    return entries_;
}

}  // namespace agentari::text::hierarchical
