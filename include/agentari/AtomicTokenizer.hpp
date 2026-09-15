#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace agentari::text::hierarchical {

using AtomicId = std::uint8_t;

enum class AtomicKind : std::uint8_t {
    character,
    punctuation,
    number,
    whitespace,
    other,
};

struct AtomicToken {
    AtomicId id{0U};
    AtomicKind kind{AtomicKind::other};
    std::size_t byte_offset{0U};
};

struct WordSpan {
    std::size_t begin{0U};
    std::size_t end{0U};
    std::string text;
};

class AtomicTokenizer {
public:
    [[nodiscard]] std::vector<AtomicToken> encode(std::string_view input) const;
    [[nodiscard]] std::string decode(std::span<const AtomicToken> tokens) const;
    [[nodiscard]] std::vector<WordSpan> word_spans(std::string_view input) const;

    [[nodiscard]] static AtomicKind classify(AtomicId value) noexcept;
    [[nodiscard]] static bool is_word_character(AtomicId value) noexcept;
};

}  // namespace agentari::text::hierarchical
