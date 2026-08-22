#include "ft/cpu_backend.h"
#include "ft/quant.h"

#include <cstring>
#include <sstream>
#include <algorithm>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ft {

namespace {

int detect_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 4;
}

std::string cpu_brand() {
#if defined(_MSC_VER)
    int regs[12];
    __cpuid(regs, 0x80000000);
    if ((uint32_t)regs[0] >= 0x80000004u) {
        __cpuid(regs + 0, 0x80000002);
        __cpuid(regs + 4, 0x80000003);
        __cpuid(regs + 8, 0x80000004);
        char brand[49] = {0};
        std::memcpy(brand, regs, sizeof(regs));
        return std::string(brand);
    }
#endif
    return "generic-x64";
}

} // namespace

void parallel_for(int64_t n, int threads, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    int nt = threads == 0 ? detect_threads() : threads;
    nt = std::max(1, std::min<int>(nt, (int)std::min<int64_t>(n, 256)));
    if (nt == 1) {
        fn(0, n);
        return;
    }
    int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve((size_t)nt - 1);
    for (int t = 1; t < nt; ++t) {
        int64_t b = t * chunk;
        if (b >= n) break;
        int64_t e = std::min(n, b + chunk);
        workers.emplace_back([&fn, b, e] { fn(b, e); });
    }
    fn(0, std::min(n, chunk));
    for (auto& w : workers) w.join();
}

void matmul_down_axpy(const float* x, int64_t M, int64_t n_in,
                      const HostTensor& W, float* y) {
    const int64_t N = W.cols();
    const DType dt = W.dtype;
    const size_t stride = W.row_stride_bytes();
    std::vector<float> row((size_t)N);
    for (int64_t i = 0; i < n_in; ++i) {
        dequant_row(row.data(), W.data + stride * (size_t)i, dt, N);
        for (int64_t m = 0; m < M; ++m) {
            const float v = x[(size_t)(m * n_in + i)];
            if (v == 0.f) continue;
            float* ym = y + m * N;
            for (int64_t j = 0; j < N; ++j) ym[j] += v * row[j];
        }
    }
}

CpuBackend::CpuBackend(int threads)
    : threads_(threads > 0 ? threads : detect_threads()) {
    std::ostringstream os;
    os << "CPU:" << cpu_brand() << " x" << threads_;
    name_ = os.str();
    info_.name = name_;
    info_.integrated = false;
    info_.uma = true;
    info_.mem_bytes = (uint64_t)threads_ * 0;
}

CpuBackend::~CpuBackend() = default;

void CpuBackend::matmul(const float* x, int64_t M, int64_t K,
                        const HostTensor& W, float* y) {
    const int64_t N = W.rows();
    const DType dt = W.dtype;
    const size_t stride = W.row_stride_bytes();
    parallel_for(N, threads_, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            const uint8_t* wrow = W.data + stride * (size_t)r;
            float acc = dot_row_f32(x, wrow, dt, K);
            for (int64_t m = 1; m < M; ++m)
                acc += dot_row_f32(x + m * K, wrow, dt, K);
            y[r] = acc;
        }
    });
}

void CpuBackend::run_expert_chunk(const ExpertChunkJob& job) {
    const int64_t M = job.M;
    const int64_t K = job.K;
    const int64_t IC = job.IC;
    if (IC <= 0 || M <= 0 || K <= 0) return;

    const uint8_t* gate_base = (const uint8_t*)job.gate;
    const uint8_t* up_base = (const uint8_t*)job.up;

    std::vector<float> h((size_t)(M * IC));

    parallel_for(M * IC, threads_, [&](int64_t b, int64_t e) {
        for (int64_t idx = b; idx < e; ++idx) {
            const int64_t m = idx / IC;
            const int64_t i = idx % IC;
            float g = dot_row_f32(job.x + m * K,
                                  gate_base + job.gate_up_row_stride * (size_t)i,
                                  job.dt, K);
            float u = dot_row_f32(job.x + m * K,
                                  up_base + job.gate_up_row_stride * (size_t)i,
                                  job.dt, K);
            g = g / (1.f + std::exp(-g));
            h[idx] = g * u;
        }
    });

    parallel_for(K, threads_, [&](int64_t kb, int64_t ke) {
        std::vector<float> row((size_t)K);
        for (int64_t m = 0; m < M; ++m) {
            const float* hm = h.data() + m * IC;
            float* out_m = job.out + m * K;
            for (int64_t i = 0; i < IC; ++i) {
                dequant_row(row.data(),
                            (const uint8_t*)job.down +
                                job.down_row_stride * (size_t)i,
                            job.dt, K);
                const float v = hm[i];
                if (v == 0.f) continue;
                for (int64_t k = kb; k < ke; ++k)
                    out_m[k] += v * row[(size_t)k];
            }
        }
    });
}

} // namespace ft
