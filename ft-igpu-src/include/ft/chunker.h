#pragma once

#include "ft/backend.h"

#include <chrono>
#include <algorithm>
#include <cstring>

namespace ft {

struct StreamerConfig {
    int64_t chunk_neurons = 0;
    int64_t staging_budget = 16ll << 20;
};

struct StreamerStats {
    uint64_t chunks_done = 0;
    uint64_t bytes_staged = 0;
    double stage_ms = 0;
    double compute_ms = 0;
};

class ExpertStreamer {
public:
    ExpertStreamer(IBackend& dev, const StreamerConfig& cfg)
        : dev_(dev), cfg_(cfg) {}

    ~ExpertStreamer() {
        if (staging_) dev_.free_staging(staging_);
    }

    void run(const float* x, int64_t M, int64_t K,
             const HostTensor& gate, const HostTensor& up,
             const HostTensor& down, float* out) {
        if (M <= 0 || K <= 0) return;
        const int64_t n_ff = gate.rows();
        int64_t C = cfg_.chunk_neurons > 0
                        ? cfg_.chunk_neurons
                        : auto_chunk(K, gate.dtype, cfg_.staging_budget);
        C -= C % QK;
        if (C <= 0) C = QK;
        C = std::min(C, n_ff);

        const size_t gu_stride = gate.row_stride_bytes();
        const DType dt = gate.dtype;

        ensure_staging(C, K, dt);

        for (int64_t i0 = 0; i0 < n_ff; i0 += C) {
            const int64_t ic = std::min(n_ff - i0, C);
            ExpertChunkJob job;
            job.x = x;
            job.out = out;
            job.dt = dt;
            job.M = M;
            job.K = K;
            job.IC = ic;
            job.gate_up_row_stride = gu_stride;

            double t = now();
            if (staging_) {
                stage_window(i0, ic, gate, up, down);
                job.gate = win_gate_p_;
                job.up = win_up_p_;
                job.down = win_down_p_;
            } else {
                job.gate = gate.data + gu_stride * (size_t)i0;
                job.up = up.data + gu_stride * (size_t)i0;
                job.down = down.data + down.row_stride_bytes() * (size_t)i0;
            }
            job.down_row_stride = down.row_stride_bytes();
            stage_ms_ += now() - t;

            t = now();
            dev_.run_expert_chunk(job);
            compute_ms_ += now() - t;
            stats_.chunks_done++;
        }
    }

    static int64_t auto_chunk(int64_t K, DType dt, int64_t budget) {
        const size_t bb = dtype_bytes_per_block(dt);
        const size_t gstride = dtype_row_bytes(dt, K);
        const double per_neuron =
            2.0 * (double)gstride + (double)K * (double)bb / 32.0;
        int64_t c = (int64_t)((double)budget / per_neuron);
        c -= c % QK;
        return std::max<int64_t>(QK, c);
    }

    StreamerStats stats() const {
        StreamerStats s = stats_;
        s.bytes_staged = bytes_staged_;
        s.stage_ms = stage_ms_;
        s.compute_ms = compute_ms_;
        return s;
    }

private:
    static size_t block_bytes(DType t) { return dtype_bytes_per_block(t); }

    static double now() {
        using namespace std::chrono;
        return duration<double, std::milli>(
                   steady_clock::now().time_since_epoch())
            .count();
    }

    static size_t window_bytes(int64_t ic, int64_t K, DType dt) {
        return (size_t)ic * dtype_row_bytes(dt, K) * 2 +
               (size_t)K * ((size_t)(ic / QK) * block_bytes(dt));
    }

    bool ensure_staging(int64_t C, int64_t K, DType dt) {
        const size_t need = window_bytes(C, K, dt);
        void* p = dev_.alloc_staging(need);
        if (!p) return false;
        staging_ = p;
        staging_cap_ = need;
        win_gate_p_ = (uint8_t*)p;
        win_up_p_ = win_gate_p_ + (size_t)C * dtype_row_bytes(dt, K);
        win_down_p_ = win_up_p_ + (size_t)C * dtype_row_bytes(dt, K);
        return true;
    }

    void stage_window(int64_t i0, int64_t ic, const HostTensor& gate,
                      const HostTensor& up, const HostTensor& down) {
        const size_t gu_rows = (size_t)ic * gate.row_stride_bytes();
        std::memcpy(win_gate_p_, gate.data + gate.row_stride_bytes() * (size_t)i0,
                    gu_rows);
        std::memcpy(win_up_p_, up.data + up.row_stride_bytes() * (size_t)i0,
                    gu_rows);

        const size_t bb = block_bytes(down.dtype);
        const int64_t b0 = i0 / QK;
        const int64_t nb = ic / QK;
        win_down_stride_ = (size_t)nb * bb;
        const int64_t K = down.rows();
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t* src = down.data +
                                 down.row_stride_bytes() * (size_t)k +
                                 (size_t)b0 * bb;
            std::memcpy(win_down_p_ + win_down_stride_ * (size_t)k, src,
                        win_down_stride_);
        }
        bytes_staged_ += gu_rows * 2 + win_down_stride_ * (size_t)K;
    }

    IBackend& dev_;
    StreamerConfig cfg_;
    StreamerStats stats_;
    void* staging_ = nullptr;
    size_t staging_cap_ = 0;
    uint8_t* win_gate_p_ = nullptr;
    uint8_t* win_up_p_ = nullptr;
    uint8_t* win_down_p_ = nullptr;
    size_t win_down_stride_ = 0;
    uint64_t bytes_staged_ = 0;
    double stage_ms_ = 0;
    double compute_ms_ = 0;
};

} // namespace ft
