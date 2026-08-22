#pragma once

#include "ft/types.h"

namespace ft {

struct ExpertChunkJob {
    const float* x = nullptr;
    float* out = nullptr;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    DType dt = DType::F32;
    int64_t M = 0;
    int64_t K = 0;
    int64_t IC = 0;
    size_t gate_up_row_stride = 0;
    size_t down_row_stride = 0;
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual std::string name() const = 0;
    virtual DeviceInfo info() const = 0;
    virtual bool ready() const = 0;

    virtual void matmul(const float* x, int64_t M, int64_t K,
                        const HostTensor& W, float* y) = 0;

    virtual void run_expert_chunk(const ExpertChunkJob& job) = 0;

    virtual void* alloc_staging(size_t bytes) { (void)bytes; return nullptr; }
    virtual void free_staging(void* p) { (void)p; }
    virtual void stage(void* dst, const void* src, size_t bytes) {
        (void)dst; (void)src; (void)bytes;
    }
    virtual void sync() {}
};

}
