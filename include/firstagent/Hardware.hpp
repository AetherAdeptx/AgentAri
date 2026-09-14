#pragma once

#include <cstddef>
#include <string_view>

#include "firstagent/System.hpp"

namespace firstagent::hardware {

struct CpuFeatures {
    std::size_t logical_processors{0U};
    std::size_t physical_processors{0U};
    std::size_t usable_processors{0U};
    std::size_t cache_line_bytes{0U};
    bool sse42{false};
    bool avx{false};
    bool avx2{false};
    bool fma{false};
    bool avx512f{false};
    system::SimdLevel simd{system::SimdLevel::scalar};
};

[[nodiscard]] CpuFeatures detect_cpu_features() noexcept;
[[nodiscard]] std::string_view preferred_simd_backend() noexcept;

}  // namespace firstagent::hardware
