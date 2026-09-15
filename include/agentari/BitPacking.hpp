#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace agentari::bits {

[[nodiscard]] constexpr bool is_power_of_two(std::size_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value,
                                             std::size_t alignment) {
    if (!is_power_of_two(alignment)) {
        throw std::invalid_argument("bit-packing alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1U;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error("aligned size overflows size_t");
    }
    return (value + mask) & ~mask;
}

[[nodiscard]] constexpr std::size_t bytes_for_bits(std::size_t bit_count) {
    return bit_count / 8U + ((bit_count % 8U) == 0U ? 0U : 1U);
}

[[nodiscard]] constexpr unsigned bits_required(std::uint64_t maximum_value) noexcept {
    unsigned bits = 0U;
    do {
        ++bits;
        maximum_value >>= 1U;
    } while (maximum_value != 0U);
    return bits;
}

[[nodiscard]] constexpr bool fits(std::uint64_t value, unsigned width) noexcept {
    return width == 64U || (width != 0U && width < 64U &&
                            value < (std::uint64_t{1U} << width));
}

class BitWriter {
public:
    void write(std::uint64_t value, unsigned width) {
        validate_width(width);
        if (!fits(value, width)) {
            throw std::out_of_range("value does not fit in requested bit width");
        }
        const std::size_t new_bit_count = checked_add(bit_count_, width);
        bytes_.resize(bytes_for_bits(new_bit_count), std::byte{0});
        for (unsigned bit = 0U; bit < width; ++bit) {
            if ((value & (std::uint64_t{1U} << bit)) != 0U) {
                const std::size_t offset = bit_count_ + bit;
                bytes_[offset / 8U] |=
                    static_cast<std::byte>(std::uint8_t{1U} << (offset % 8U));
            }
        }
        bit_count_ = new_bit_count;
    }

    [[nodiscard]] std::size_t bit_count() const noexcept { return bit_count_; }
    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    static void validate_width(unsigned width) {
        if (width == 0U || width > 64U) {
            throw std::invalid_argument("bit width must be between 1 and 64");
        }
    }

    static std::size_t checked_add(std::size_t left, std::size_t right) {
        if (left > std::numeric_limits<std::size_t>::max() - right) {
            throw std::overflow_error("packed bit count overflows size_t");
        }
        return left + right;
    }

    std::vector<std::byte> bytes_;
    std::size_t bit_count_{0U};
};

class BitReader {
public:
    BitReader(std::span<const std::byte> bytes, std::size_t bit_count)
        : bytes_(bytes), bit_count_(bit_count) {
        if (bytes.size() > std::numeric_limits<std::size_t>::max() / 8U ||
            bit_count > bytes.size() * 8U) {
            throw std::invalid_argument("packed bit count exceeds storage");
        }
    }

    [[nodiscard]] std::uint64_t read(unsigned width) {
        if (width == 0U || width > 64U || bit_offset_ > bit_count_ ||
            static_cast<std::size_t>(width) > bit_count_ - bit_offset_) {
            throw std::out_of_range("packed bit read exceeds storage");
        }
        std::uint64_t result = 0U;
        for (unsigned bit = 0U; bit < width; ++bit) {
            const std::size_t offset = bit_offset_ + bit;
            const std::uint8_t byte = std::to_integer<std::uint8_t>(bytes_[offset / 8U]);
            if ((byte & (std::uint8_t{1U} << (offset % 8U))) != 0U) {
                result |= std::uint64_t{1U} << bit;
            }
        }
        bit_offset_ += width;
        return result;
    }

    [[nodiscard]] std::size_t remaining_bits() const noexcept {
        return bit_count_ - bit_offset_;
    }

private:
    std::span<const std::byte> bytes_;
    std::size_t bit_count_{0U};
    std::size_t bit_offset_{0U};
};

class PackedVector {
public:
    PackedVector() = default;
    PackedVector(unsigned width, std::size_t size)
        : width_(width), size_(size) {
        if (width == 0U || width > 64U) {
            throw std::invalid_argument("packed vector width must be between 1 and 64");
        }
        bytes_.assign(bytes_for_bits(bit_count()), std::byte{0});
    }

    void set(std::size_t index, std::uint64_t value) {
        check_index(index);
        if (!fits(value, width_)) {
            throw std::out_of_range("packed vector value does not fit its width");
        }
        const std::size_t offset = checked_offset(index);
        for (unsigned bit = 0U; bit < width_; ++bit) {
            const std::size_t byte_offset = (offset + bit) / 8U;
            const std::byte mask = static_cast<std::byte>(
                std::uint8_t{1U} << ((offset + bit) % 8U));
            if ((value & (std::uint64_t{1U} << bit)) != 0U) {
                bytes_[byte_offset] |= mask;
            } else {
                bytes_[byte_offset] &= ~mask;
            }
        }
    }

    [[nodiscard]] std::uint64_t get(std::size_t index) const {
        check_index(index);
        const std::size_t offset = checked_offset(index);
        std::uint64_t result = 0U;
        for (unsigned bit = 0U; bit < width_; ++bit) {
            const std::uint8_t byte = std::to_integer<std::uint8_t>(
                bytes_[(offset + bit) / 8U]);
            if ((byte & (std::uint8_t{1U} << ((offset + bit) % 8U))) != 0U) {
                result |= std::uint64_t{1U} << bit;
            }
        }
        return result;
    }

    [[nodiscard]] unsigned width() const noexcept { return width_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    [[nodiscard]] std::size_t bit_count() const {
        if (size_ != 0U && static_cast<std::size_t>(width_) >
                               std::numeric_limits<std::size_t>::max() / size_) {
            throw std::overflow_error("packed vector size overflows size_t");
        }
        return static_cast<std::size_t>(width_) * size_;
    }

    [[nodiscard]] std::size_t checked_offset(std::size_t index) const {
        if (index != 0U && static_cast<std::size_t>(width_) >
                               std::numeric_limits<std::size_t>::max() / index) {
            throw std::overflow_error("packed vector offset overflows size_t");
        }
        return static_cast<std::size_t>(width_) * index;
    }

    void check_index(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("packed vector index is outside its size");
        }
    }

    unsigned width_{0U};
    std::size_t size_{0U};
    std::vector<std::byte> bytes_;
};

// A cache-line-aware packed array for hot categorical/state data. Unlike
// PackedVector, which is a compact byte stream, this layout starts every
// block on an AlignmentBytes boundary. That costs up to one padded block but
// keeps unrelated worker ranges from sharing cache lines and makes block
// partitioning explicit for parallel scans.
template <unsigned BitsPerValue, std::size_t AlignmentBytes = 64U>
class AlignedPackedArray {
    static_assert(BitsPerValue > 0U && BitsPerValue <= 64U,
                  "aligned packed values require 1..64 bits");
    static_assert(AlignmentBytes >= alignof(std::uint64_t) &&
                      is_power_of_two(AlignmentBytes),
                  "aligned packed storage requires a power-of-two alignment");

    struct alignas(AlignmentBytes) Block {
        std::array<std::byte, AlignmentBytes> bytes{};
    };

public:
    AlignedPackedArray() = default;

    explicit AlignedPackedArray(std::size_t size) {
        resize(size);
    }

    void resize(std::size_t size) {
        if (size != 0U && static_cast<std::size_t>(BitsPerValue) >
                              std::numeric_limits<std::size_t>::max() / size) {
            throw std::overflow_error("aligned packed array size overflows size_t");
        }
        const std::size_t bytes = bytes_for_bits(static_cast<std::size_t>(BitsPerValue) * size);
        const std::size_t blocks = bytes == 0U ? 0U :
            (bytes - 1U) / AlignmentBytes + 1U;
        blocks_.assign(blocks, Block{});
        size_ = size;
    }

    void clear() noexcept {
        for (Block& block : blocks_) {
            block.bytes.fill(std::byte{0});
        }
    }

    void set(std::size_t index, std::uint64_t value) {
        check_index(index);
        if (!fits(value, BitsPerValue)) {
            throw std::out_of_range("aligned packed value does not fit its width");
        }
        write_bits(index * static_cast<std::size_t>(BitsPerValue), value);
    }

    [[nodiscard]] std::uint64_t get(std::size_t index) const {
        check_index(index);
        return read_bits(index * static_cast<std::size_t>(BitsPerValue));
    }

    [[nodiscard]] constexpr unsigned width() const noexcept { return BitsPerValue; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t block_count() const noexcept { return blocks_.size(); }
    [[nodiscard]] constexpr std::size_t alignment_bytes() const noexcept {
        return AlignmentBytes;
    }
    [[nodiscard]] std::size_t bytes_reserved() const noexcept {
        return blocks_.size() * AlignmentBytes;
    }

    // Exposes cache-line-sized read ranges without exposing bit-level writes.
    [[nodiscard]] std::span<const std::byte> block_bytes(std::size_t block) const {
        if (block >= blocks_.size()) {
            throw std::out_of_range("aligned packed block is outside its size");
        }
        return std::span<const std::byte>(blocks_[block].bytes);
    }

private:
    void check_index(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("aligned packed array index is outside its size");
        }
    }

    void write_bits(std::size_t bit_offset, std::uint64_t value) {
        for (unsigned bit = 0U; bit < BitsPerValue; ++bit) {
            const std::size_t offset = bit_offset + bit;
            Block& block = blocks_[offset / (AlignmentBytes * 8U)];
            const std::size_t block_bit = offset % (AlignmentBytes * 8U);
            const std::size_t byte = block_bit / 8U;
            const std::uint8_t mask = std::uint8_t{1U} << (block_bit % 8U);
            std::uint8_t current = std::to_integer<std::uint8_t>(block.bytes[byte]);
            if ((value & (std::uint64_t{1U} << bit)) != 0U) {
                current = static_cast<std::uint8_t>(current | mask);
            } else {
                current = static_cast<std::uint8_t>(current & static_cast<std::uint8_t>(~mask));
            }
            block.bytes[byte] = static_cast<std::byte>(current);
        }
    }

    [[nodiscard]] std::uint64_t read_bits(std::size_t bit_offset) const {
        std::uint64_t result = 0U;
        for (unsigned bit = 0U; bit < BitsPerValue; ++bit) {
            const std::size_t offset = bit_offset + bit;
            const Block& block = blocks_[offset / (AlignmentBytes * 8U)];
            const std::size_t block_bit = offset % (AlignmentBytes * 8U);
            const std::uint8_t current = std::to_integer<std::uint8_t>(
                block.bytes[block_bit / 8U]);
            if ((current & (std::uint8_t{1U} << (block_bit % 8U))) != 0U) {
                result |= std::uint64_t{1U} << bit;
            }
        }
        return result;
    }

    std::size_t size_{0U};
    std::vector<Block> blocks_;
};

}  // namespace agentari::bits
