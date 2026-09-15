#include "agentari/Hardware.hpp"

namespace agentari::hardware {

CpuFeatures detect_cpu_features() noexcept {
    const system::CpuCapabilities& detected = system::cpu_capabilities();
    CpuFeatures result;
    result.logical_processors = detected.topology.logical_processors;
    result.physical_processors = detected.topology.physical_processors;
    result.usable_processors = detected.topology.usable_processors;
    result.cache_line_bytes = detected.topology.cache_line_bytes;
    result.sse42 = detected.sse42;
    result.avx = detected.avx;
    result.avx2 = detected.avx2;
    result.fma = detected.fma;
    result.avx512f = detected.avx512f;
    result.simd = detected.simd;
    return result;
}

std::string_view preferred_simd_backend() noexcept {
    return system::simd_name(system::cpu_capabilities().simd);
}

}  // namespace agentari::hardware
