#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <stdexcept>
#include <functional>

namespace ft {

using token_t = int32_t;

enum class DType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    BF16 = 30,
};

constexpr size_t QK = 32;

inline size_t dtype_elems_per_block(DType t) {
    switch (t) {
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q8_0:
            return QK;
        default:
            return 1;
    }
}

inline size_t dtype_bytes_per_block(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::Q4_0: return 18;
        case DType::Q4_1: return 20;
        case DType::Q8_0: return 34;
        default: throw std::runtime_error("unknown dtype");
    }
}

inline size_t dtype_row_bytes(DType t, int64_t n_cols) {
    size_t eb = dtype_elems_per_block(t);
    size_t bb = dtype_bytes_per_block(t);
    return (size_t)((n_cols + (int64_t)eb - 1) / (int64_t)eb) * bb;
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {
            int e = -1;
            uint32_t m = man;
            do { ++e; m <<= 1; } while (!(m & 0x400u));
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x007FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFF) {
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x00800000u;
        int shift = (int)(14 - exp);
        uint32_t half_man = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        if ((rem > (uint32_t)(1 << (shift - 1))) ||
            (rem == (uint32_t)(1 << (shift - 1)) && (half_man & 1))) {
            half_man++;
        }
        return (uint16_t)(sign | half_man);
    }
    uint32_t hr = man >> 13;
    uint32_t remn = man & 0x1FFFu;
    if (remn > 0x1000u || (remn == 0x1000u && (hr & 1))) hr++;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | hr);
}

inline float bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct HostTensor {
    std::string name;
    int64_t ne[4] = {1, 1, 1, 1};
    int32_t n_dims = 0;
    DType dtype = DType::F32;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    int64_t cols() const { return ne[0]; }
    int64_t rows() const { return ne[1]; }
    int64_t dim2() const { return ne[2]; }
    int64_t dim3() const { return ne[3]; }

    size_t row_stride_bytes() const { return dtype_row_bytes(dtype, ne[0]); }

    const uint8_t* slice(int64_t i2, int64_t i3 = 0) const {
        size_t plane = row_stride_bytes() * (size_t)ne[1];
        return data + plane * ((size_t)i3 * (size_t)ne[2] + (size_t)i2);
    }

    size_t slice_bytes() const { return row_stride_bytes() * (size_t)ne[1]; }
};

struct DeviceInfo {
    std::string name;
    bool integrated = false;
    bool uma = false;
    uint64_t mem_bytes = 0;
};

} // namespace ft
