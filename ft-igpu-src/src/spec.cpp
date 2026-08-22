#include "ft/spec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ft {

namespace {
uint64_t rng_next(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
float rng_float(uint64_t& s) { return (float)(rng_next(s) >> 40) * (1.f / 16777216.f); }
} // namespace

NgramDrafter::NgramDrafter(int max_n) : max_n_(std::max(1, max_n)) {
    cache_.resize((size_t)max_n_);
}

void NgramDrafter::feed(const std::vector<token_t>& toks) {
    const int64_t n = (int64_t)toks.size();
    if ((size_t)n <= fed_) return;
    for (int64_t i = (int64_t)fed_; i < n; ++i) {
        for (int order = 1; order <= max_n_; ++order) {
            if (i - order < 0 || i + order > n) continue;
            std::vector<token_t> key(toks.begin() + (i - order),
                                     toks.begin() + i);
            auto& cont = cache_[(size_t)(order - 1)][key];
            if (cont.size() < 8 &&
                (cont.empty() || cont.back() != toks[(size_t)i])) {
                cont.push_back(toks[(size_t)i]);
            }
        }
    }
    fed_ = (size_t)n;
}

std::vector<token_t> NgramDrafter::draft(const std::vector<token_t>& ctx,
                                         int k, std::vector<float>* probs) {
    st_.lookups++;
    if (probs) probs->clear();
    const int64_t n = (int64_t)ctx.size();
    if (n < 1) return {};
    feed(ctx);
    const int64_t hi = std::min<int64_t>(max_n_, n);
    for (int64_t order = hi; order >= 1; --order) {
        std::vector<token_t> key(ctx.end() - order, ctx.end());
        auto& m = cache_[(size_t)(order - 1)];
        auto it = m.find(key);
        if (it == m.end()) continue;
        st_.hits++;
        std::vector<token_t> out;
        for (token_t t : it->second) {
            if ((int)out.size() >= k) break;
            out.push_back(t);
        }
        return out;
    }
    return {};
}

NgramDrafter::Stats NgramDrafter::stats() const {
    Stats s = st_;
    for (const auto& m : cache_) s.entries += m.size();
    return s;
}

DraftModelDrafter::DraftModelDrafter(const EngineOptions& opts)
    : e_(new Engine) {
    e_->load(opts);
}

DraftModelDrafter::~DraftModelDrafter() = default;

void DraftModelDrafter::sync(const std::vector<token_t>& ctx) {
    if (e_->npast() > (int64_t)ctx.size()) e_->reset();
    const int64_t have = e_->npast();
    if (have < (int64_t)ctx.size()) {
        std::vector<token_t> tail(ctx.begin() + have, ctx.end());
        e_->prefill(tail);
    }
    synced_ = ctx.size();
}

float softmax_prob(const float* logits, int64_t V, token_t t) {
    float mx = logits[0];
    for (int64_t i = 1; i < V; ++i) mx = std::max(mx, logits[i]);
    double sum = 0;
    for (int64_t i = 0; i < V; ++i) sum += std::exp((double)(logits[i] - mx));
    return (float)(std::exp((double)(logits[(size_t)t] - mx)) / sum);
}

std::vector<token_t> DraftModelDrafter::draft(const std::vector<token_t>& ctx,
                                              int k,
                                              std::vector<float>* probs) {
    sync(ctx);
    pending_base_ = ctx.size();
    const int64_t V = e_->cfg().vocab;
    std::vector<token_t> out;
    if (probs) probs->clear();
    for (int j = 0; j < k; ++j) {
        const float* lg = e_->logits();
        token_t t = Sampler::argmax(lg, V);
        float p = softmax_prob(lg, V, t);
        out.push_back(t);
        if (probs) probs->push_back(p);
        if (j + 1 < k) e_->step(t);
    }
    return out;
}

void DraftModelDrafter::on_commit(int accepted) {
    e_->truncate_to((int64_t)pending_base_ + accepted);
    synced_ = pending_base_ + (size_t)accepted;
}

DFlashBlockDrafter::DFlashBlockDrafter(const EngineOptions& opts, int rounds)
    : e_(new Engine), rounds_(std::max(1, rounds)) {
    e_->load(opts);
}

DFlashBlockDrafter::~DFlashBlockDrafter() = default;

void DFlashBlockDrafter::sync(const std::vector<token_t>& ctx) {
    if (e_->npast() > (int64_t)ctx.size()) e_->reset();
    const int64_t have = e_->npast();
    if (have < (int64_t)ctx.size()) {
        std::vector<token_t> tail(ctx.begin() + have, ctx.end());
        e_->prefill(tail);
    }
    synced_ = ctx.size();
}

std::vector<token_t> DFlashBlockDrafter::draft(const std::vector<token_t>& ctx,
                                               int k,
                                               std::vector<float>* probs) {
    sync(ctx);
    const int64_t V = e_->cfg().vocab;
    std::vector<float> base(e_->logits(), e_->logits() + V);
    std::vector<token_t> g((size_t)k, 0);
    g[0] = Sampler::argmax(base.data(), V);
    for (int r = 1; r < k; ++r) g[(size_t)r] = g[0];

    for (int round = 0; round < rounds_; ++round) {
        std::vector<float> rows = e_->probe_rows(g, -1);
        for (int j = 1; j < k; ++j)
            g[(size_t)j] =
                Sampler::argmax(rows.data() + (size_t)((j - 1) * V), V);
    }

    if (probs) {
        probs->assign((size_t)k, 0.f);
        (*probs)[0] = softmax_prob(base.data(), V, g[0]);
        std::vector<float> rows = e_->probe_rows(g, -1);
        for (int j = 1; j < k; ++j)
            (*probs)[(size_t)j] =
                softmax_prob(rows.data() + (size_t)((j - 1) * V), V,
                             g[(size_t)j]);
    }
    return g;
}

void MultiDrafter::add(std::unique_ptr<IDrafter> d) {
    parts_.push_back(std::move(d));
}

bool MultiDrafter::provides_probs() const {
    for (const auto& p : parts_)
        if (p->provides_probs()) return true;
    return false;
}

std::vector<token_t> MultiDrafter::draft(const std::vector<token_t>& ctx,
                                         int k, std::vector<float>* probs) {
    if (probs) probs->clear();
    std::vector<token_t> out;
    std::unordered_set<token_t> seen;
    for (auto& d : parts_) {
        const bool pp_ok = d->provides_probs();
        std::vector<float> pp;
        std::vector<token_t> part = d->draft(ctx, k, pp_ok ? &pp : nullptr);
        for (size_t i = 0; i < part.size() && (int)out.size() < k; ++i) {
            if (seen.insert(part[i]).second) {
                out.push_back(part[i]);
                if (probs) {
                    if (pp_ok && i < pp.size()) probs->push_back(pp[i]);
                    else probs->push_back(-1.f);
                }
            }
        }
    }
    return out;
}

void MultiDrafter::on_commit(int accepted) {
    for (auto& d : parts_) d->on_commit(accepted);
}

int accept_chain_greedy(const float* base_row, const float* rows, int64_t V,
                        const token_t* props, int64_t k) {
    int a = 0;
    while (a < k) {
        const float* pred = (a == 0) ? base_row : rows + (size_t)((a - 1) * V);
        if (Sampler::argmax(pred, V) != props[a]) break;
        ++a;
    }
    return a;
}

int accept_chain_sampling(const float* base_row, const float* rows,
                          const float* pd, int64_t V, const token_t* props,
                          int64_t k, float, uint64_t& rng) {
    auto prob_of = [&](const float* lg, token_t t) {
        float mx = lg[0];
        for (int64_t i = 1; i < V; ++i) mx = std::max(mx, lg[i]);
        double sum = 0;
        for (int64_t i = 0; i < V; ++i) sum += std::exp((double)(lg[i] - mx));
        return (float)(std::exp((double)(lg[(int)t] - mx)) / sum);
    };
    int a = 0;
    while (a < k) {
        const float* pred = (a == 0) ? base_row : rows + (size_t)((a - 1) * V);
        if (pd[a] <= 0.f) {
            if (Sampler::argmax(pred, V) != props[a]) break;
        } else {
            const float pt = prob_of(pred, props[a]);
            const float pdr = pd[a];
            const float ratio = std::min(1.f, pt / std::max(pdr, 1e-9f));
            if (rng_float(rng) > ratio) break;
        }
        ++a;
    }
    return a;
}

GenResult generate(Engine& eng, IDrafter* drafter, const SampleParams& sp,
                   int max_new_tokens, int spec_k,
                   const std::function<void(token_t, const std::string&)>&
                       on_token) {
    GenResult res;
    const int64_t V = eng.cfg().vocab;
    const token_t eos = eng.tok().eos();
    Sampler samp;
    samp.reset(sp.seed);

    auto emit = [&](token_t t) {
        ++res.produced;
        if (on_token) on_token(t, eng.tok().decode({t}, false));
    };

    std::vector<float> prev_logits;
    if (const float* lg = eng.logits())
        prev_logits.assign(lg, lg + V);

    uint64_t rsamp = sp.seed ? sp.seed : 88172645463325252ull;

    while (res.produced < max_new_tokens) {
        if (!drafter) {
            const float* lg = prev_logits.empty() ? eng.logits() : prev_logits.data();
            token_t t = samp.sample(lg, V, sp);
            emit(t);
            if (t == eos) { res.hit_eos = true; break; }
            const float* nl = eng.step(t);
            prev_logits.assign(nl, nl + V);
            continue;
        }

        std::vector<float> probs;
        std::vector<token_t> props =
            drafter->draft(eng.history(), spec_k, &probs);

        if (!probs.empty()) {
            size_t cut = props.size();
            double cum = 1.0;
            for (size_t i = 0; i < props.size(); ++i) {
                if (probs[i] >= 0 && probs[i] < 0.02f) { cut = i; break; }
                cum *= (probs[i] > 0 ? probs[i] : 1.f);
                if (cum < 0.05f) { cut = i + 1; break; }
            }
            props.resize(cut);
        }

        if (props.empty()) {
            const float* lg = prev_logits.empty() ? eng.logits() : prev_logits.data();
            token_t t = samp.sample(lg, V, sp);
            emit(t);
            if (t == eos) { res.hit_eos = true; break; }
            const float* nl = eng.step(t);
            prev_logits.assign(nl, nl + V);
            drafter->on_commit(0);
            continue;
        }

        prev_logits.assign(eng.logits(), eng.logits() + V);
        const int64_t np_before = eng.npast();
        auto view = eng.extend_verify(props);

        int acc = drafter->provides_probs()
                      ? accept_chain_sampling(prev_logits.data(), view.rows,
                                              probs.data(), V, props.data(),
                                              view.k, sp.temp, rsamp)
                      : accept_chain_greedy(prev_logits.data(), view.rows, V,
                                            props.data(), view.k);

        const int64_t bonus_idx = std::min<int64_t>(acc, view.k - 1);
        const float* bonus_row = view.rows + (size_t)(bonus_idx * V);
        token_t bonus = samp.sample(bonus_row, V, sp);

        eng.truncate_to(np_before + acc);
        for (int i = 0; i < acc; ++i) {
            emit(props[(size_t)i]);
            if (props[(size_t)i] == eos) { res.hit_eos = true; }
        }
        if (res.hit_eos) {
            drafter->on_commit(acc);
            break;
        }
        emit(bonus);
        if (bonus == eos) {
            const float* nl = eng.step(bonus);
            prev_logits.assign(nl, nl + V);
            drafter->on_commit(acc);
            res.hit_eos = true;
            break;
        }
        const float* nl = eng.step(bonus);
        prev_logits.assign(nl, nl + V);
        drafter->on_commit(acc);
        eng.add_spec((uint64_t)props.size(), (uint64_t)acc, 1);
    }
    return res;
}

} // namespace ft
