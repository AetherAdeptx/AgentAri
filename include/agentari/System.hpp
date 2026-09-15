#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace agentari::system {

enum class SimdLevel : std::uint8_t {
    scalar,
    sse42,
    avx,
    avx2_fma,
    avx512f,
};

struct CpuTopology {
    std::size_t logical_processors{1U};
    std::size_t physical_processors{1U};
    std::size_t usable_processors{1U};
    std::size_t cache_line_bytes{64U};
};

struct CpuCapabilities {
    CpuTopology topology{};
    bool sse42{false};
    bool avx{false};
    bool avx2{false};
    bool fma{false};
    bool avx512f{false};
    SimdLevel simd{SimdLevel::scalar};
};

struct GpuCapabilities {
    std::size_t device_count{0U};
    std::size_t vram_bytes{0U};
    bool nvidia{false};
    bool amd{false};
    bool intel{false};
    // This is a hardware candidate report. Actual Vulkan availability still
    // belongs to the Vulkan backend because it depends on loader/driver/API
    // support, not just on a DRM device existing.
    bool vulkan_candidate{false};
};

inline std::size_t detect_usable_processors(std::size_t fallback) noexcept {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) {
            return static_cast<std::size_t>(count);
        }
    }
#endif
    return std::max<std::size_t>(1U, fallback);
}

inline std::size_t detect_physical_processors(const CpuTopology& topology) noexcept {
#if defined(__linux__)
    try {
        std::set<std::pair<int, int>> cores;
        for (std::size_t cpu = 0U; cpu < topology.logical_processors; ++cpu) {
            bool usable = true;
            cpu_set_t affinity;
            CPU_ZERO(&affinity);
            if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
                usable = cpu < CPU_SETSIZE && CPU_ISSET(static_cast<int>(cpu), &affinity);
            }
            if (!usable) {
                continue;
            }
            const std::filesystem::path base =
                std::filesystem::path("/sys/devices/system/cpu") /
                ("cpu" + std::to_string(cpu)) / "topology";
            std::ifstream package_file(base / "physical_package_id");
            std::ifstream core_file(base / "core_id");
            int package = 0;
            int core = static_cast<int>(cpu);
            if (!(package_file >> package) || !(core_file >> core)) {
                continue;
            }
            cores.emplace(package, core);
        }
        if (!cores.empty()) {
            return cores.size();
        }
    } catch (...) {
        // Hardware probing is optional. The usable logical count is safer than
        // failing startup because sysfs is restricted or unavailable.
    }
#endif
    return std::max<std::size_t>(1U, std::min(topology.logical_processors,
                                              topology.usable_processors));
}

inline CpuCapabilities detect_cpu_capabilities() noexcept {
    CpuCapabilities result;
    const std::size_t hardware_threads =
        static_cast<std::size_t>(std::thread::hardware_concurrency());
    result.topology.logical_processors = std::max<std::size_t>(1U, hardware_threads);
    result.topology.usable_processors = detect_usable_processors(
        result.topology.logical_processors);
    result.topology.physical_processors = detect_physical_processors(result.topology);

#if defined(__cpp_lib_hardware_interference_size)
    result.topology.cache_line_bytes = std::hardware_destructive_interference_size;
#endif
    if (result.topology.cache_line_bytes == 0U) {
        result.topology.cache_line_bytes = 64U;
    }

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    result.sse42 = __builtin_cpu_supports("sse4.2") != 0;
    result.avx = __builtin_cpu_supports("avx") != 0;
    result.avx2 = __builtin_cpu_supports("avx2") != 0;
    result.fma = __builtin_cpu_supports("fma") != 0;
    result.avx512f = __builtin_cpu_supports("avx512f") != 0;
#endif
#endif

    if (result.avx512f) {
        result.simd = SimdLevel::avx512f;
    } else if (result.avx2 && result.fma) {
        result.simd = SimdLevel::avx2_fma;
    } else if (result.avx) {
        result.simd = SimdLevel::avx;
    } else if (result.sse42) {
        result.simd = SimdLevel::sse42;
    }
    return result;
}

inline GpuCapabilities detect_gpu_capabilities() noexcept {
    GpuCapabilities result;
#if defined(__linux__)
    try {
        const std::filesystem::path drm_root("/sys/class/drm");
        std::error_code filesystem_error;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(drm_root, filesystem_error)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("card", 0U) != 0U || name.size() <= 4U ||
                name.find('-', 4U) != std::string::npos ||
                name[4U] < '0' || name[4U] > '9') {
                continue;
            }
            ++result.device_count;
            std::ifstream vendor_file(entry.path() / "device/vendor");
            std::string vendor;
            vendor_file >> vendor;
            if (vendor == "0x10de") {
                result.nvidia = true;
            } else if (vendor == "0x1002") {
                result.amd = true;
            } else if (vendor == "0x8086") {
                result.intel = true;
            }
            std::ifstream memory_file(entry.path() / "device/mem_info_vram_total");
            std::uint64_t memory = 0U;
            if (memory_file >> memory && memory <=
                                          std::numeric_limits<std::size_t>::max() -
                                              result.vram_bytes) {
                result.vram_bytes += static_cast<std::size_t>(memory);
            }
        }
        result.vulkan_candidate = result.device_count != 0U;
    } catch (...) {
        // A container or restricted sysfs should not prevent CPU execution.
    }
#endif
    return result;
}

inline const CpuCapabilities& cpu_capabilities() noexcept {
    static const CpuCapabilities capabilities = detect_cpu_capabilities();
    return capabilities;
}

inline const GpuCapabilities& gpu_capabilities() noexcept {
    static const GpuCapabilities capabilities = detect_gpu_capabilities();
    return capabilities;
}

inline const char* simd_name(SimdLevel level) noexcept {
    switch (level) {
    case SimdLevel::avx512f:
        return "AVX-512F";
    case SimdLevel::avx2_fma:
        return "AVX2+FMA";
    case SimdLevel::avx:
        return "AVX";
    case SimdLevel::sse42:
        return "SSE4.2";
    case SimdLevel::scalar:
    default:
        return "scalar";
    }
}

}  // namespace agentari::system
