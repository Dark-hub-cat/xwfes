#pragma once

#include "ft/engine.h"

#include <functional>
#include <unordered_set>

namespace ft {

enum class SpecMode { None, Ngram, DraftModel, DFlash, Mlsd };

struct SpecParams {
    SpecMode mode = SpecMode::None;
    int k = 4;
    float p_min = 0.05f;
    float p_split = 0.9f;
    std::string draft_model_path;
    int dflash_rounds = 3;
};

class IDrafter {
public:
    virtual ~IDrafter() = default;
    virtual std::vector<token_t> draft(const std::vector<token_t>& ctx, int k,
                                       std::vector<float>* probs) = 0;
    virtual void on_commit(int accepted) { (void)accepted; }
    virtual const char* name() const { return "drafter"; }
    virtual bool provides_probs() const { return false; }
};

class NgramDrafter : public IDrafter {
public:
    explicit NgramDrafter(int max_n = 4);
    std::vector<token_t> draft(const std::vector<token_t>& ctx, int k,
                               std::vector<float>* probs) override;
    void feed(const std::vector<token_t>& toks);
    struct Stats {
        uint64_t lookups = 0;
        uint64_t hits = 0;
        uint64_t entries = 0;
    };
    Stats stats() const;

private:
    struct VectorHash {
        size_t operator()(const std::vector<token_t>& v) const {
            size_t h = 1469598103934665603ull;
            for (token_t t : v) {
                h ^= (size_t)(uint32_t)t + 0x9e3779b97f4a7c15ull;
                h *= 1099511628211ull;
            }
            return h;
        }
    };
    using Map =
        std::unordered_map<std::vector<token_t>, std::vector<token_t>,
                           VectorHash>;
    int max_n_;
    size_t fed_ = 0;
    std::vector<Map> cache_;
    mutable Stats st_{};
};

class DraftModelDrafter : public IDrafter {
public:
    explicit DraftModelDrafter(const EngineOptions& opts);
    ~DraftModelDrafter() override;
    std::vector<token_t> draft(const std::vector<token_t>& ctx, int k,
                               std::vector<float>* probs) override;
    void on_commit(int accepted) override;
    const char* name() const override { return "draft-model"; }
    bool provides_probs() const override { return true; }

private:
    void sync(const std::vector<token_t>& ctx);
    std::unique_ptr<Engine> e_;
    size_t synced_ = 0;
    size_t pending_base_ = 0;
};

class DFlashBlockDrafter : public IDrafter {
public:
    DFlashBlockDrafter(const EngineOptions& opts, int rounds);
    ~DFlashBlockDrafter() override;
    std::vector<token_t> draft(const std::vector<token_t>& ctx, int k,
                               std::vector<float>* probs) override;
    const char* name() const override { return "dflash-block"; }
    bool provides_probs() const override { return true; }

private:
    void sync(const std::vector<token_t>& ctx);
    std::unique_ptr<Engine> e_;
    int rounds_;
    size_t synced_ = 0;
};

class MultiDrafter : public IDrafter {
public:
    void add(std::unique_ptr<IDrafter> d);
    std::vector<token_t> draft(const std::vector<token_t>& ctx, int k,
                               std::vector<float>* probs) override;
    void on_commit(int accepted) override;
    const char* name() const override { return "multi"; }
    bool provides_probs() const override;
    size_t count() const { return parts_.size(); }
    const std::unique_ptr<IDrafter>& part(size_t i) const { return parts_[i]; }

private:
    std::vector<std::unique_ptr<IDrafter>> parts_;
};

int accept_chain_greedy(const float* base_row, const float* rows, int64_t V,
                        const token_t* props, int64_t k);

int accept_chain_sampling(const float* base_row, const float* rows,
                          const float* pd, int64_t V, const token_t* props,
                          int64_t k, float temp, uint64_t& rng);

struct GenResult {
    int produced = 0;
    bool hit_eos = false;
};

GenResult generate(Engine& eng, IDrafter* drafter, const SampleParams& sp,
                   int max_new_tokens, int spec_k,
                   const std::function<void(token_t, const std::string&)>&
                       on_token);

} // namespace ft
