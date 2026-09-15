#pragma once

#include "agentari/AtomicTokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace agentari::text::hierarchical {

using WordId = std::uint32_t;

struct WordEntry {
    WordId id{0U};
    std::string text;
    std::size_t frequency_rank{0U};
    std::uint64_t observations{0U};
};

class WordVocabulary {
public:
    explicit WordVocabulary(std::size_t capacity = 100000U);

    // Words are added by observation. A frequency-ranked file can be layered
    // on later without changing the atomic or learned-part interfaces.
    [[nodiscard]] WordId observe_word(std::string_view word,
                                       std::size_t frequency_rank = 0U);
    void observe_text(std::string_view input);

    [[nodiscard]] WordId lookup(std::string_view word) const noexcept;
    [[nodiscard]] std::string_view text(WordId id) const noexcept;
    [[nodiscard]] const WordEntry* entry(WordId id) const noexcept;
    [[nodiscard]] std::vector<WordId> encode_text(std::string_view input) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] const std::vector<WordEntry>& entries() const noexcept;

    static constexpr WordId unknown_id = 0U;

private:
    std::size_t capacity_;
    std::unordered_map<std::string, WordId> ids_;
    std::vector<WordEntry> entries_;

    [[nodiscard]] static std::string normalize(std::string_view word);
};

}  // namespace agentari::text::hierarchical
