#pragma once

#include "ft/backend.h"
#include "ft/cpu_backend.h"

#ifdef FT_HAVE_VULKAN
#include "ft/vulkan_backend.h"
#endif

#include <memory>
#include <vector>
#include <algorithm>

namespace ft {

enum class IgpuMode { Auto, On, Off };

class HeteroScheduler {
public:
    HeteroScheduler(std::shared_ptr<CpuBackend> cpu,
                    std::shared_ptr<IBackend> igpu,
                    IgpuMode mode)
        : cpu_(std::move(cpu)), igpu_(std::move(igpu)) {
        igpu_usable_ = false;
        if (mode == IgpuMode::On && !igpu_) {
            throw std::runtime_error("--igpu on: integrated GPU not available");
        }
        if (mode == IgpuMode::Auto && igpu_) igpu_usable_ = true;
        if (mode == IgpuMode::On) igpu_usable_ = true;
        user_set_ = mode != IgpuMode::Auto;
    }

    bool igpu_enabled() const { return igpu_usable_; }

    void set_igpu_enabled(bool on) {
        if (on && !igpu_) throw std::runtime_error("iGPU not available on this system");
        igpu_usable_ = on;
        user_set_ = true;
    }

    IBackend& expert_device(int64_t est_chunk_bytes) {
        if (igpu_enabled() && igpu_->ready() &&
            est_chunk_bytes <= (int64_t)igpu_->info().mem_bytes / 4) {
            return *igpu_;
        }
        return *cpu_;
    }

    IBackend& dense_device() { return igpu_enabled() ? *igpu_ : *cpu_; }

    std::string describe() const {
        std::string s = "[plan] devices:";
        s += "\n  cpu   : " + cpu_->name();
        if (igpu_) {
            s += "\n  igpu  : " + igpu_->name() +
                 " (mem " + std::to_string(igpu_->info().mem_bytes >> 20) + " MB)" +
                 " state=" + (igpu_enabled() ? "enabled" : "disabled");
        } else {
            s += "\n  igpu  : none";
        }
        return s;
    }

private:
    std::shared_ptr<CpuBackend> cpu_;
    std::shared_ptr<IBackend> igpu_;
    bool igpu_usable_ = false;
    bool user_set_ = false;
};

} // namespace ft
