#pragma once

#include "ft/backend.h"

#include <thread>
#include <vector>
#include <functional>

namespace ft {

void parallel_for(int64_t n, int threads, const std::function<void(int64_t, int64_t)>& fn);

void matmul_down_axpy(const float* x, int64_t M, int64_t n_in,
                      const HostTensor& W, float* y);

class CpuBackend : public IBackend {
public:
    explicit CpuBackend(int threads);
    ~CpuBackend() override;

    std::string name() const override { return name_; }
    DeviceInfo info() const override { return info_; }
    bool ready() const override { return true; }

    void matmul(const float* x, int64_t M, int64_t K,
                const HostTensor& W, float* y) override;

    void run_expert_chunk(const ExpertChunkJob& job) override;

private:
    int threads_;
    std::string name_;
    DeviceInfo info_;
    std::vector<float> h_buf_;
};

} // namespace ft
