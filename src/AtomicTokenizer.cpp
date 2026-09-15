#include "agentari/AtomicTokenizer.hpp"

namespace agentari::text::hierarchical {

AtomicKind AtomicTokenizer::classify(const AtomicId value) noexcept {
    if (is_word_character(value)) {
        return AtomicKind::character;
    }
    if (value >= static_cast<AtomicId>('0') && value <= static_cast<AtomicId>('9')) {
        return AtomicKind::number;
    }
    if (value == static_cast<AtomicId>(' ') || value == static_cast<AtomicId>('\t') ||
        value == static_cast<AtomicId>('\n') || value == static_cast<AtomicId>('\r')) {
        return AtomicKind::whitespace;
    }
    if ((value >= 33U && value <= 47U) || (value >= 58U && value <= 64U) ||
        (value >= 91U && value <= 96U) || (value >= 123U && value <= 126U)) {
        return AtomicKind::punctuation;
    }
    return AtomicKind::other;
}

bool AtomicTokenizer::is_word_character(const AtomicId value) noexcept {
    return (value >= static_cast<AtomicId>('A') && value <= static_cast<AtomicId>('Z')) ||
           (value >= static_cast<AtomicId>('a') && value <= static_cast<AtomicId>('z'));
}

std::vector<AtomicToken> AtomicTokenizer::encode(const std::string_view input) const {
    std::vector<AtomicToken> result;
    result.reserve(input.size());
    for (std::size_t index = 0U; index < input.size(); ++index) {
        const AtomicId value = static_cast<AtomicId>(static_cast<unsigned char>(input[index]));
        result.push_back(AtomicToken{value, classify(value), index});
    }
    return result;
}

std::string AtomicTokenizer::decode(const std::span<const AtomicToken> tokens) const {
    std::string result;
    result.reserve(tokens.size());
    for (const AtomicToken token : tokens) {
        result.push_back(static_cast<char>(token.id));
    }
    return result;
}

std::vector<WordSpan> AtomicTokenizer::word_spans(const std::string_view input) const {
    std::vector<WordSpan> result;
    std::size_t begin = 0U;
    while (begin < input.size()) {
        while (begin < input.size() && !is_word_character(static_cast<AtomicId>(
                   static_cast<unsigned char>(input[begin])))) {
            ++begin;
        }
        if (begin == input.size()) {
            break;
        }
        std::size_t end = begin;
        while (end < input.size() && is_word_character(static_cast<AtomicId>(
                                      static_cast<unsigned char>(input[end])))) {
            ++end;
        }
        result.push_back(WordSpan{begin, end, std::string(input.substr(begin, end - begin))});
        begin = end;
    }
    return result;
}

}  // namespace agentari::text::hierarchical
