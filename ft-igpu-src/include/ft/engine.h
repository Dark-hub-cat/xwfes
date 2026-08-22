#pragma once

#include "ft/graph.h"
#include "ft/kv_cache.h"
#include "ft/sampler.h"
#include "ft/scheduler.h"
#include "ft/chunker.h"
#include "ft/throttle.h"
#include "ft/tokenizer.h"
#include "ft/cpu_backend.h"

namespace ft {

struct EngineOptions {
    std::string model_path;
    int threads = 0;
    IgpuMode igpu = IgpuMode::Auto;
    int64_t chunk_neurons = 0;
    int64_t staging_budget = 16ll << 20;
    int64_t ctx_len = 0;
    int64_t prefill_batch = 256;
    ThrottleConfig throttle{};
};

struct RunStats {
    StreamerStats streamer{};
    uint64_t prefill_tokens = 0;
    uint64_t gen_tokens = 0;
    double prefill_ms = 0;
    double gen_ms = 0;
    uint64_t spec_proposed = 0;
    uint64_t spec_accepted = 0;
    uint64_t spec_cycles = 0;

    double gen_tps() const { return gen_ms > 0 ? gen_tokens * 1000.0 / gen_ms : 0; }
    double prefill_tps() const {
        return prefill_ms > 0 ? prefill_tokens * 1000.0 / prefill_ms : 0;
    }
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void load(const EngineOptions& opt);

    const ModelConfig& cfg() const;
    const Tokenizer& tok() const;
    int64_t ctx_cap() const;
    int64_t npast() const;
    const std::vector<token_t>& history() const;
    RunStats stats() const;

    void prefill(const std::vector<token_t>& toks);
    const float* step(token_t t);
    const float* logits() const;

    std::vector<float> probe_rows(const std::vector<token_t>& toks,
                                  int layer_limit = -1);

    struct VerifyView {
        const float* rows;
        int64_t k;
        token_t base_token;
    };
    VerifyView extend_verify(const std::vector<token_t>& proposals);

    void truncate_to(int64_t n);
    void reset();

    void add_spec(uint64_t proposed, uint64_t accepted, uint64_t cycles);
    HeteroScheduler& sched();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    const float* forward(const std::vector<token_t>& toks, bool write_kv,
                         int layer_limit);
    void ensure_capacity(int64_t add);
    friend class DraftModelDrafter;
    friend class DFlashBlockDrafter;
};

} // namespace ft
