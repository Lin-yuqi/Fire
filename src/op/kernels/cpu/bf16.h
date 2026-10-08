#pragma once

#include <cstdint>
#include <cstring>

namespace kernel {

inline float bf16_to_fp32(uint16_t value) {
    // BF16 stores the upper 16 bits of an IEEE 754 FP32 value.
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace kernel
