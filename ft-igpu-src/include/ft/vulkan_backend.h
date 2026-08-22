#pragma once

#include "ft/backend.h"
#include "ft/throttle.h"

namespace ft {

class VulkanBackend : public IBackend {
public:
    VulkanBackend(bool prefer_integrated, ThrottleConfig throttle);
    ~VulkanBackend() override;

    std::string name() const override;
    DeviceInfo info() const override;
    bool ready() const override;

    void matmul(const float* x, int64_t M, int64_t K,
                const HostTensor& W, float* y) override;

    void run_expert_chunk(const ExpertChunkJob& job) override;

    void* alloc_staging(size_t bytes) override;
    void free_staging(void* p) override;
    void stage(void* dst, const void* src, size_t bytes) override;
    void sync() override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace ft
