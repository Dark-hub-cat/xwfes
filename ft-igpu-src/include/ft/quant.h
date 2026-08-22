#pragma once

#include "ft/types.h"

namespace ft {

float dot_block_window(const float* vals, const uint8_t* row, DType t,
                       int64_t off_elems, int64_t n);

inline void dequant_row(float* dst, const uint8_t* src, DType t, int64_t n) {
    switch (t) {
        case DType::F32:
            std::memcpy(dst, src, (size_t)n * 4);
            return;
        case DType::F16: {
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                dst[i] = half_to_float(h);
            }
            return;
        }
        case DType::BF16: {
            for (int64_t i = 0; i < n; ++i) {
                uint16_t b;
                std::memcpy(&b, src + i * 2, 2);
                dst[i] = bf16_to_float(b);
            }
            return;
        }
        case DType::Q8_0: {
            const int64_t nb = ((n + (int64_t)QK - 1) / (int64_t)QK);
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t* p = src + (size_t)b * 34;
                uint16_t dh;
                std::memcpy(&dh, p, 2);
                float d = half_to_float(dh);
                int64_t base = b * (int64_t)QK;
                int64_t cnt = std::min<int64_t>((int64_t)QK, n - base);
                for (int64_t j = 0; j < cnt; ++j)
                    dst[base + j] = d * (float)(int8_t)p[2 + j];
            }
            return;
        }
        case DType::Q4_0: {
            const int64_t nb = ((n + (int64_t)QK - 1) / (int64_t)QK);
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t* p = src + (size_t)b * 18;
                uint16_t dh;
                std::memcpy(&dh, p, 2);
                float d = half_to_float(dh);
                int64_t base = b * (int64_t)QK;
                int64_t cnt = std::min<int64_t>((int64_t)QK, n - base);
                for (int64_t j = 0; j < cnt; ++j) {
                    uint8_t byte = p[2 + (j >> 1)];
                    int nib = (j & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    dst[base + j] = d * (float)(nib - 8);
                }
            }
            return;
        }
        case DType::Q4_1: {
            const int64_t nb = ((n + (int64_t)QK - 1) / (int64_t)QK);
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t* p = src + (size_t)b * 20;
                uint16_t dmh, dnh;
                std::memcpy(&dmh, p, 2);
                std::memcpy(&dnh, p + 2, 2);
                float dm = half_to_float(dmh);
                float dn = half_to_float(dnh);
                int64_t base = b * (int64_t)QK;
                int64_t cnt = std::min<int64_t>((int64_t)QK, n - base);
                for (int64_t j = 0; j < cnt; ++j) {
                    uint8_t byte = p[4 + (j >> 1)];
                    int nib = (j & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    dst[base + j] = dm + dn * (float)nib;
                }
            }
            return;
        }
        default:
            throw std::runtime_error("unsupported dtype");
    }
}

inline float dot_row_f32(const float* x, const uint8_t* w, DType t, int64_t n) {
    if (t == DType::F32) {
        const float* wf = (const float*)w;
        float acc = 0.f;
        for (int64_t i = 0; i < n; ++i) acc += x[i] * wf[i];
        return acc;
    }
    alignas(64) float buf[2048];
    float acc = 0.f;
    int64_t i = 0;
    while (i < n) {
        int64_t chunk = std::min<int64_t>(n - i, 2048);
        dequant_row(buf, w + dtype_row_bytes(t, i), t, chunk);
        for (int64_t j = 0; j < chunk; ++j) acc += x[i + j] * buf[j];
        i += chunk;
    }
    return acc;
}

inline float dot_block_window(const float* vals, const uint8_t* row, DType t,
                              int64_t off_elems, int64_t n) {
    if (t == DType::F32) {
        const float* wf = (const float*)row + off_elems;
        float acc = 0.f;
        for (int64_t i = 0; i < n; ++i) acc += vals[i] * wf[i];
        return acc;
    }
    if (t == DType::F16 || t == DType::BF16) {
        const size_t es = (t == DType::F16) ? 2 : 2;
        float acc = 0.f;
        for (int64_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, row + (size_t)(off_elems + i) * es, 2);
            float w = (t == DType::F16) ? half_to_float(h) : bf16_to_float(h);
            acc += vals[i] * w;
        }
        return acc;
    }
    alignas(64) float buf[QK];
    const int64_t startb = off_elems / (int64_t)QK;
    const int64_t endb = ((int64_t)(off_elems + n) - 1) / (int64_t)QK;
    const size_t bb = (t == DType::Q8_0) ? 34 : ((t == DType::Q4_0) ? 18 : 20);
    float acc = 0.f;
    int64_t consumed = 0;
    for (int64_t b = startb; b <= endb; ++b) {
        const uint8_t* bp = row + (size_t)b * bb;
        int64_t lo = std::max((int64_t)off_elems, b * (int64_t)QK);
        int64_t hi = std::min((int64_t)(off_elems + n), (b + 1) * (int64_t)QK);
        if (lo >= hi) continue;
        dequant_row(buf, bp, t, QK);
        for (int64_t i = lo; i < hi; ++i) {
            acc += vals[consumed + (i - lo)] * buf[i - b * QK];
        }
        consumed += hi - lo;
    }
    return acc;
}

} // namespace ft
