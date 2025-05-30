#pragma once

#include <cstdint>

// Define half type for CPU code if not using CUDA
#ifndef __CUDACC__
typedef struct alignas(2) {
    uint16_t x;
} half;
#else
#include <cuda_fp16.h>
#endif

namespace alphazero {
namespace mcts {

// CPU-friendly half precision conversion functions
inline float half_to_float(const half& h) {
#ifdef __CUDACC__
    return __half2float(h);
#else
    uint16_t h_bits = *reinterpret_cast<const uint16_t*>(&h);
    uint32_t sign = (h_bits & 0x8000) << 16;
    uint32_t exp = ((h_bits & 0x7c00) >> 10);
    uint32_t mant = (h_bits & 0x03ff) << 13;
    
    // Handle special cases
    if (exp == 0) {
        // Zero or denormal
        return 0.0f;
    } else if (exp == 0x1f) {
        // Inf or NaN
        exp = 0xff;
    } else {
        // Normal number
        exp = exp + 112;
    }
    
    union { uint32_t i; float f; } converter;
    converter.i = sign | (exp << 23) | mant;
    return converter.f;
#endif
}

inline half float_to_half(float f) {
#ifdef __CUDACC__
    return __float2half(f);
#else
    union { float f; uint32_t i; } converter;
    converter.f = f;
    
    uint16_t sign = (converter.i >> 16) & 0x8000;
    int exp = ((converter.i >> 23) & 0xff) - 112;
    uint16_t mant = (converter.i >> 13) & 0x3ff;
    
    // Handle special cases
    if (exp <= 0) {
        // Too small - flush to zero
        exp = 0;
        mant = 0;
    } else if (exp >= 0x1f) {
        // Too large - clamp to infinity
        exp = 0x1f;
        mant = 0;
    }
    
    uint16_t h_bits = sign | (exp << 10) | mant;
    half h;
    *reinterpret_cast<uint16_t*>(&h) = h_bits;
    return h;
#endif
}

} // namespace mcts
} // namespace alphazero