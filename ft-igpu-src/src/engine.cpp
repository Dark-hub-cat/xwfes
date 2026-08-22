#include "ft/engine.h"
#include "ft/quant.h"

#include <chrono>
#include <cstring>
#include <algorithm>

#ifdef FT_HAVE_VULKAN
#include "ft/vulkan_backend.h"
#endif

namespace ft {

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
               steady_clock::now().time_since_epoch())
        .count();
}

void rms_norm(float* out, const float* x, const HostTensor* w, int64_t n,
              float eps) {
    float ss = 0.f;
    for (int64_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.f / std::sqrt(ss / (float)n + eps);
    if (!w) {
        for (int64_t i = 0; i < n; ++i) out[i] = x[i] * inv;
        return;
    }
    if (w->dtype == DType::F32) {
        const float* wf = (const float*)w->data;
        for (int64_t i = 0; i < n; ++i) out[i] = x[i] * inv * wf[i];
        return;
    }
    static thread_local std::vector<float> wb;
    wb.resize((size_t)n);
    dequant_row(wb.data(), w->data, w->dtype, n);
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * inv * wb[(size_t)i];
}

void rope_vec(float* v, int64_t hd, int64_t pos, double theta, bool neox) {
    for (int64_t i = 0; i < hd / 2; ++i) {
        const double f = std::pow(theta, -2.0 * (double)i / (double)hd);
        const double ang = (double)pos * f;
        const float c = (float)std::cos(ang);
        const float s = (float)std::sin(ang);
        if (neox) {
            const int64_t j = i + hd / 2;
            const float a = v[i], b = v[j];
            v[i] = a * c - b * s;
            v[j] = a * s + b * c;
        } else {
            const float a = v[2 * i], b = v[2 * i + 1];
            v[2 * i] = a * c - b * s;
            v[2 * i + 1] = a * s + b * c;
        }
    }
}

void softmax_inplace(float* p, int64_t n) {
    float mx = p[0];
    for (int64_t i = 1; i < n; ++i) mx = std::max(mx, p[i]);
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        p[i] = std::exp(p[i] - mx);
        sum += p[i];
    }
    const float inv = (float)(1.0 / sum);
    for (int64_t i = 0; i < n; ++i) p[i] *= inv;
}

HostTensor expert_view(const HostTensor& t, int64_t e) {
    HostTensor v = t;
    v.data = t.slice(e);
    v.n_dims = 2;
    v.ne[2] = 1;
    v.ne[3] = 1;
    return v;
}

} // namespace

struct Engine::Impl {
    EngineOptions opt;
    std::unique_ptr<Gguf> gg;
    ModelConfig cfg;
    Tokenizer tokz;

    std::shared_ptr<CpuBackend> cpu;
    std::shared_ptr<IBackend> acc;
    std::unique_ptr<HeteroScheduler> sched;
    std::unique_ptr<ExpertStreamer> streamer;
    bool streamer_on_acc = false;
    DeviceGovernor gov{ThrottleConfig{}};

    KvCache kv;
    PrefixCache prefix{32, 8};
    int64_t ctx_cap = 0;
    std::vector<token_t> history;
    std::vector<float> rows;
    int64_t rows_m = 0;
    RunStats st;

    struct Layer {
        const HostTensor* attn_norm = nullptr;
        const HostTensor* q = nullptr;
        const HostTensor* k = nullptr;
        const HostTensor* v = nullptr;
        const HostTensor* o = nullptr;
        const HostTensor* qn = nullptr;
        const HostTensor* kn = nullptr;
        const HostTensor* post = nullptr;
        const HostTensor* gate_inp = nullptr;
        const HostTensor* gexps = nullptr;
        const HostTensor* uexps = nullptr;
        const HostTensor* dexps = nullptr;
        const HostTensor* sgate = nullptr;
        const HostTensor* sup = nullptr;
        const HostTensor* sdown = nullptr;
        const HostTensor* fgate = nullptr;
        const HostTensor* fup = nullptr;
        const HostTensor* fdown = nullptr;
        std::vector<float> qb, kb, vb, rbias;
    };
    std::vector<Layer> ly;
    const HostTensor* embd = nullptr;
    const HostTensor* out_w = nullptr;
    const HostTensor* out_norm = nullptr;

    std::vector<float> x, xn, q, kb2, vb2, attout, tmpo, h1, h2, hrow, router,
        xe, acb, top_w;

    IBackend& dense_dev() { return sched->dense_device(); }

    void attn(int64_t l, int64_t base, int64_t M, const float* kblk,
              const float* vblk) {
        const ModelConfig& c = cfg;
        const int64_t hd = c.head_dim;
        const int64_t nh = c.n_head;
        const int64_t dk = c.n_head_kv * hd;
        const int64_t gqa = nh / c.n_head_kv;
        const float scale = 1.f / std::sqrt((float)hd);
        parallel_for(M * nh, opt.threads, [&](int64_t b, int64_t e) {
            std::vector<float> sc;
            for (int64_t idx = b; idx < e; ++idx) {
                const int64_t m = idx / nh;
                const int64_t h = idx % nh;
                const int64_t kvh = h / gqa;
                const float* qseg = &q[(size_t)(m * nh + h) * hd];
                const int64_t len = base + m + 1;
                sc.resize((size_t)len);
                for (int64_t p = 0; p < len; ++p) {
                    const float* kp =
                        (p < base ? kv.k(l) + (size_t)(p * dk)
                                  : kblk + (size_t)((p - base) * dk)) +
                        kvh * hd;
                    float d = 0.f;
                    for (int64_t t = 0; t < hd; ++t) d += qseg[t] * kp[t];
                    sc[(size_t)p] = d * scale;
                }
                softmax_inplace(sc.data(), len);
                float* oseg = &attout[(size_t)(m * nh + h) * hd];
                for (int64_t t = 0; t < hd; ++t) oseg[t] = 0.f;
                for (int64_t p = 0; p < len; ++p) {
                    const float wgt = sc[(size_t)p];
                    if (wgt == 0.f) continue;
                    const float* vp =
                        (p < base ? kv.v(l) + (size_t)(p * dk)
                                  : vblk + (size_t)((p - base) * dk)) +
                        kvh * hd;
                    for (int64_t t = 0; t < hd; ++t) oseg[t] += wgt * vp[t];
                }
            }
        });
    }

    void moe_ffn(const Layer& T, int64_t M) {
        const ModelConfig& c = cfg;
        const int64_t K = c.n_embd;
        const int64_t E = c.n_experts;
        const int64_t U = std::min(c.n_expert_used, E);
        IBackend& dev = dense_dev();
        const bool gov_on = acc && (&dev == acc.get());

        router.resize((size_t)(M * E));
        if (gov_on) gov.before_submit();
        dev.matmul(xn.data(), M, K, *T.gate_inp, router.data());
        if (gov_on) gov.after_submit();

        std::vector<int64_t> idxv((size_t)E);
        std::vector<int64_t> used;
        std::vector<std::vector<std::pair<int64_t, int64_t>>> tok_of_e(
            (size_t)E);
        top_w.assign((size_t)(M * U), 0.f);

        for (int64_t m = 0; m < M; ++m) {
            float* r = router.data() + m * E;
            if (!T.rbias.empty())
                for (int64_t j = 0; j < E; ++j) r[j] += T.rbias[(size_t)j];
            softmax_inplace(r, E);
            for (int64_t j = 0; j < E; ++j) idxv[(size_t)j] = j;
            std::partial_sort(idxv.begin(), idxv.begin() + U, idxv.end(),
                              [&](int64_t a, int64_t b) { return r[a] > r[b]; });
            float wsum = 0.f;
            for (int64_t u = 0; u < U; ++u) wsum += r[idxv[(size_t)u]];
            const float inv = wsum > 0 ? 1.f / wsum : 0.f;
            for (int64_t u = 0; u < U; ++u) {
                const int64_t e = idxv[(size_t)u];
                top_w[(size_t)(m * U + u)] = r[e] * inv;
                tok_of_e[(size_t)e].emplace_back(m, u);
                if (tok_of_e[(size_t)e].size() == 1) used.push_back(e);
            }
        }

        for (int64_t e : used) {
            auto& list = tok_of_e[(size_t)e];
            const int64_t S = (int64_t)list.size();
            xe.resize((size_t)(S * K));
            for (int64_t si = 0; si < S; ++si)
                std::memcpy(&xe[(size_t)(si * K)],
                            &xn[(size_t)(list[(size_t)si].first * K)],
                            (size_t)K * sizeof(float));
            acb.assign((size_t)(S * K), 0.f);
            HostTensor ge = expert_view(*T.gexps, e);
            HostTensor ue = expert_view(*T.uexps, e);
            HostTensor de = expert_view(*T.dexps, e);
            if (gov_on && streamer_on_acc) gov.before_submit();
            streamer->run(xe.data(), S, K, ge, ue, de, acb.data());
            if (gov_on && streamer_on_acc) gov.after_submit();
            for (int64_t si = 0; si < S; ++si) {
                const int64_t m = list[(size_t)si].first;
                const int64_t u = list[(size_t)si].second;
                const float wgt = top_w[(size_t)(m * U + u)];
                const float* arow = &acb[(size_t)(si * K)];
                float* trow = tmpo.data() + (size_t)(m * K);
                for (int64_t j = 0; j < K; ++j) trow[j] += wgt * arow[j];
            }
            list.clear();
        }

        if (T.sgate && T.sup && T.sdown) {
            const int64_t FS = T.sgate->rows();
            h1.resize((size_t)(M * FS));
            h2.resize((size_t)(M * FS));
            if (gov_on) gov.before_submit();
            dev.matmul(xn.data(), M, K, *T.sgate, h1.data());
            dev.matmul(xn.data(), M, K, *T.sup, h2.data());
            if (gov_on) gov.after_submit();
            for (int64_t i = 0; i < M * FS; ++i) {
                float g = h1[(size_t)i];
                g = g / (1.f + std::exp(-g));
                h1[(size_t)i] = g * h2[(size_t)i];
            }
            matmul_down_axpy(h1.data(), M, FS, *T.sdown, tmpo.data());
        }
    }

    void dense_ffn(const Layer& T, int64_t M) {
        const ModelConfig& c = cfg;
        const int64_t K = c.n_embd;
        const int64_t F = c.n_ff;
        IBackend& dev = dense_dev();
        h1.resize((size_t)(M * F));
        h2.resize((size_t)(M * F));
        hrow.resize((size_t)(M * F));
        dev.matmul(xn.data(), M, K, *T.fgate, h1.data());
        dev.matmul(xn.data(), M, K, *T.fup, h2.data());
        for (int64_t i = 0; i < M * F; ++i) {
            float g = h1[(size_t)i];
            g = g / (1.f + std::exp(-g));
            hrow[(size_t)i] = g * h2[(size_t)i];
        }
        matmul_down_axpy(hrow.data(), M, F, *T.fdown, tmpo.data());
    }
};

Engine::Engine() : impl_(new Impl) {}
Engine::~Engine() = default;

void Engine::load(const EngineOptions& opt) {
    Impl& I = *impl_;
    I.opt = opt;
    I.gg.reset(new Gguf(opt.model_path));
    I.cfg = config_from_gguf(*I.gg);
    I.tokz = Tokenizer::from_gguf(*I.gg);
    Weights w(*I.gg);
    ModelConfig& c = I.cfg;

    I.cpu.reset(new CpuBackend(opt.threads));
#ifdef FT_HAVE_VULKAN
    if (opt.igpu != IgpuMode::Off) {
        try {
            I.acc.reset(new VulkanBackend(true, opt.throttle));
        } catch (...) {
            I.acc.reset();
        }
    }
#endif
    I.sched.reset(new HeteroScheduler(I.cpu, I.acc, opt.igpu));
    I.gov.set_config(opt.throttle);

    I.ctx_cap = opt.ctx_len > 0 ? opt.ctx_len
                                : std::min<int64_t>(c.ctx_train, 8192);
    const int64_t dk = c.n_head_kv * c.head_dim;
    I.kv.alloc(c.n_layer, I.ctx_cap + opt.prefill_batch + 16, dk, dk);

    StreamerConfig sc;
    sc.chunk_neurons = opt.chunk_neurons;
    sc.staging_budget = opt.staging_budget;
    I.streamer.reset(
        new ExpertStreamer(I.sched->expert_device(opt.staging_budget), sc));
    I.streamer_on_acc =
        I.acc && (&I.sched->expert_device(opt.staging_budget) == I.acc.get());

    I.ly.resize((size_t)c.n_layer);
    for (int64_t l = 0; l < c.n_layer; ++l) {
        Impl::Layer& T = I.ly[(size_t)l];
        T.attn_norm = w.layer(l, {"blk.%d.attn_norm.weight",
                                  "blk.%d.input_layernorm.weight"});
        T.q = w.layer(l, {"blk.%d.attn_q.weight", "blk.%d.self_attn.q_proj.weight"});
        T.k = w.layer(l, {"blk.%d.attn_k.weight", "blk.%d.self_attn.k_proj.weight"});
        T.v = w.layer(l, {"blk.%d.attn_v.weight", "blk.%d.self_attn.v_proj.weight"});
        T.o = w.layer(l, {"blk.%d.attn_output.weight",
                          "blk.%d.self_attn.o_proj.weight"});
        T.qn = w.layer(l, {"blk.%d.attn_q_norm.weight",
                           "blk.%d.self_attn.q_norm.weight"});
        T.kn = w.layer(l, {"blk.%d.attn_k_norm.weight",
                           "blk.%d.self_attn.k_norm.weight"});
        T.post = w.layer(l, {"blk.%d.post_attention_norm.weight",
                             "blk.%d.post_attention_layernorm.weight",
                             "blk.%d.ffn_norm.weight"});
        if (!T.q || !T.k || !T.v || !T.o || !T.attn_norm || !T.post)
            throw std::runtime_error("missing attention tensors at layer " +
                                     std::to_string(l));

        auto bias_load = [&](const char* pat, std::vector<float>& dst) {
            if (const HostTensor* b = w.layer(l, {pat})) {
                dst.resize((size_t)b->rows());
                dequant_row(dst.data(), b->data, b->dtype, b->rows());
            }
        };
        bias_load("blk.%d.attn_q.bias", T.qb);
        bias_load("blk.%d.attn_k.bias", T.kb);
        bias_load("blk.%d.attn_v.bias", T.vb);

        T.gate_inp = w.layer(l, {"blk.%d.ffn_gate_inp.weight"});
        T.gexps = w.layer(l, {"blk.%d.ffn_gate_exps.weight"});
        T.uexps = w.layer(l, {"blk.%d.ffn_up_exps.weight"});
        T.dexps = w.layer(l, {"blk.%d.ffn_down_exps.weight"});
        if (!c.moe && T.gate_inp && T.gexps) c.moe = true;
        if (c.moe) {
            if (!T.gate_inp || !T.gexps || !T.uexps || !T.dexps)
                throw std::runtime_error("incomplete MoE tensors at layer " +
                                         std::to_string(l));
            if (!c.n_experts) {
                c.n_experts = T.gexps->dim2();
                c.n_expert_used = c.n_experts >= 8 ? 8 : 2;
            }
            if (!c.n_ff) c.n_ff = T.gexps->rows();
            if (const HostTensor* rb =
                    w.layer(l, {"blk.%d.exp_probs_b.bias"})) {
                T.rbias.resize((size_t)rb->rows());
                dequant_row(T.rbias.data(), rb->data, rb->dtype, rb->rows());
            }
            T.sgate = w.layer(l, {"blk.%d.ffn_gate_shexp.weight"});
            T.sup = w.layer(l, {"blk.%d.ffn_up_shexp.weight"});
            T.sdown = w.layer(l, {"blk.%d.ffn_down_shexp.weight"});
        } else {
            T.fgate = w.layer(l, {"blk.%d.ffn_gate.weight",
                                  "blk.%d.mlp.gate_proj.weight"});
            T.fup = w.layer(l, {"blk.%d.ffn_up.weight",
                                "blk.%d.mlp.up_proj.weight"});
            T.fdown = w.layer(l, {"blk.%d.ffn_down.weight",
                                  "blk.%d.mlp.down_proj.weight"});
            if (!T.fgate || !T.fup || !T.fdown)
                throw std::runtime_error("missing FFN tensors at layer " +
                                         std::to_string(l));
            if (!c.n_ff) c.n_ff = T.fgate->rows();
        }
    }
    if (!c.moe) {
        c.n_experts = 0;
        c.n_expert_used = 0;
    }

    I.embd = w.t("token_embd.weight");
    I.out_w = w.t("output.weight");
    I.out_norm = w.t("output_norm.weight");
    if (!I.embd) throw std::runtime_error("missing token_embd.weight");
    if (!I.out_norm)
        throw std::runtime_error("missing output_norm.weight");
    c.tied_embd = I.out_w == nullptr;
    if (!c.vocab) c.vocab = I.embd->rows();
}

const ModelConfig& Engine::cfg() const { return impl_->cfg; }
const Tokenizer& Engine::tok() const { return impl_->tokz; }
int64_t Engine::ctx_cap() const { return impl_->ctx_cap; }
int64_t Engine::npast() const { return impl_->kv.npast(); }
const std::vector<token_t>& Engine::history() const { return impl_->history; }

RunStats Engine::stats() const {
    RunStats s = impl_->st;
    s.streamer = impl_->streamer->stats();
    return s;
}

HeteroScheduler& Engine::sched() { return *impl_->sched; }
void Engine::add_spec(uint64_t p, uint64_t a, uint64_t cy) {
    impl_->st.spec_proposed += p;
    impl_->st.spec_accepted += a;
    impl_->st.spec_cycles += cy;
}

const float* Engine::logits() const {
    const Impl& I = *impl_;
    if (I.rows.empty() || I.rows_m <= 0) return nullptr;
    return I.rows.data() + ((size_t)I.rows_m - 1) * (size_t)I.cfg.vocab;
}

void Engine::truncate_to(int64_t n) {
    Impl& I = *impl_;
    n = std::clamp<int64_t>(n, 0, (int64_t)I.history.size());
    I.history.resize((size_t)n);
    I.kv.set_npast(n);
}

void Engine::reset() {
    Impl& I = *impl_;
    I.history.clear();
    I.kv.clear();
    I.rows.clear();
    I.rows_m = 0;
}

void Engine::ensure_capacity(int64_t add) {
    Impl& I = *impl_;
    if (npast() + add <= ctx_cap()) return;
    const int64_t blk = I.prefix.block();
    int64_t keep = ctx_cap() / 2;
    keep -= keep % blk;
    keep = std::min<int64_t>(keep, (int64_t)I.history.size());
    if (keep < blk) {
        reset();
        return;
    }
    std::vector<token_t> tail(I.history.end() - keep, I.history.end());
    reset();
    const double t0 = now_ms();
    for (size_t i = 0; i < tail.size();) {
        size_t j = std::min(tail.size(), i + (size_t)I.opt.prefill_batch);
        forward(
            std::vector<token_t>(tail.begin() + i, tail.begin() + j), true, -1);
        i = j;
    }
    I.st.prefill_ms += now_ms() - t0;
}

const float* Engine::forward(const std::vector<token_t>& toks, bool write_kv,
                             int layer_limit) {
    Impl& I = *impl_;
    const ModelConfig& c = I.cfg;
    const int64_t M = (int64_t)toks.size();
    if (M <= 0) return nullptr;
    const int64_t K = c.n_embd;
    const int64_t hd = c.head_dim;
    const int64_t nh = c.n_head;
    const int64_t nkv = c.n_head_kv;
    const int64_t dk = nkv * hd;
    const int64_t V = c.vocab;

    if (write_kv && npast() + M > ctx_cap()) {
        const int64_t blk = I.prefix.block();
        int64_t keep = ctx_cap() / 2;
        keep -= keep % blk;
        keep = std::min<int64_t>(keep, (int64_t)I.history.size());
        std::vector<token_t> tail(I.history.end() - keep, I.history.end());
        reset();
        prefill(tail);
    }

    const int64_t base = npast();
    const int64_t nl = layer_limit > 0
                           ? std::min<int64_t>((int64_t)layer_limit, c.n_layer)
                           : c.n_layer;

    I.x.resize((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m) {
        const token_t t = toks[(size_t)m];
        if (t < 0 || t >= c.vocab)
            throw std::runtime_error("token id out of range");
        std::memcpy(&I.x[(size_t)(m * K)], I.embd->slice(t),
                    (size_t)K * sizeof(float));
    }
    I.xn.resize((size_t)(M * K));
    I.q.resize((size_t)(M * nh * hd));
    I.attout.resize((size_t)(M * nh * hd));
    I.tmpo.resize((size_t)(M * K));
    I.kb2.resize((size_t)(M * dk));
    I.vb2.resize((size_t)(M * dk));

    for (int64_t l = 0; l < nl; ++l) {
        const Impl::Layer& T = I.ly[(size_t)l];
        IBackend& dev = I.dense_dev();
        const bool gov_on = I.acc && (&dev == I.acc.get());

        for (int64_t m = 0; m < M; ++m)
            rms_norm(&I.xn[(size_t)(m * K)], &I.x[(size_t)(m * K)],
                     T.attn_norm, K, c.rms_eps);

        if (gov_on) I.gov.before_submit();
        dev.matmul(I.xn.data(), M, K, *T.q, I.q.data());
        dev.matmul(I.xn.data(), M, K, *T.k, I.kb2.data());
        dev.matmul(I.xn.data(), M, K, *T.v, I.vb2.data());
        if (gov_on) I.gov.after_submit();

        auto add_bias = [&](std::vector<float>& buf,
                            const std::vector<float>& bias, int64_t dim) {
            if (bias.empty()) return;
            for (int64_t m = 0; m < M; ++m)
                for (int64_t j = 0; j < dim; ++j)
                    buf[(size_t)(m * dim + j)] += bias[(size_t)j];
        };
        add_bias(I.q, T.qb, nh * hd);
        add_bias(I.kb2, T.kb, dk);
        add_bias(I.vb2, T.vb, dk);

        if (T.qn || T.kn) {
            auto head_rms = [&](std::vector<float>& buf, int64_t heads,
                                const HostTensor* wn) {
                if (!wn) return;
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t h = 0; h < heads; ++h)
                        rms_norm(&buf[(size_t)(m * heads + h) * hd],
                                 &buf[(size_t)(m * heads + h) * hd], wn, hd,
                                 c.rms_eps);
            };
            head_rms(I.q, nh, T.qn);
            head_rms(I.kb2, nkv, T.kn);
        }

        for (int64_t m = 0; m < M; ++m) {
            float* qm = &I.q[(size_t)(m * nh * hd)];
            for (int64_t h = 0; h < nh; ++h)
                rope_vec(qm + h * hd, hd, base + m, c.rope_theta, c.rope_neox);
            float* km = &I.kb2[(size_t)(m * dk)];
            for (int64_t h = 0; h < nkv; ++h)
                rope_vec(km + h * hd, hd, base + m, c.rope_theta, c.rope_neox);
        }

        const float* kblk;
        const float* vblk;
        if (write_kv) {
            float* kd = I.kv.k(l) + (size_t)(base * dk);
            float* vd = I.kv.v(l) + (size_t)(base * dk);
            std::memcpy(kd, I.kb2.data(), (size_t)(M * dk) * sizeof(float));
            std::memcpy(vd, I.vb2.data(), (size_t)(M * dk) * sizeof(float));
            kblk = kd;
            vblk = vd;
        } else {
            kblk = I.kb2.data();
            vblk = I.vb2.data();
        }
        I.attn(l, base, M, kblk, vblk);

        if (gov_on) I.gov.before_submit();
        dev.matmul(I.attout.data(), M, nh * hd, *T.o, I.tmpo.data());
        if (gov_on) I.gov.after_submit();
        for (int64_t i = 0; i < M * K; ++i) I.x[(size_t)i] += I.tmpo[(size_t)i];

        for (int64_t m = 0; m < M; ++m)
            rms_norm(&I.xn[(size_t)(m * K)], &I.x[(size_t)(m * K)], T.post, K,
                     c.rms_eps);

        std::fill(I.tmpo.begin(), I.tmpo.end(), 0.f);
        if (c.moe) I.moe_ffn(T, M);
        else I.dense_ffn(T, M);
        for (int64_t i = 0; i < M * K; ++i) I.x[(size_t)i] += I.tmpo[(size_t)i];
    }

    for (int64_t m = 0; m < M; ++m)
        rms_norm(&I.xn[(size_t)(m * K)], &I.x[(size_t)(m * K)], I.out_norm, K,
                 c.rms_eps);
    const HostTensor* lm = I.out_w ? I.out_w : I.embd;
    I.rows.resize((size_t)(M * V));
    I.dense_dev().matmul(I.xn.data(), M, K, *lm, I.rows.data());
    I.rows_m = M;

    if (write_kv) {
        I.kv.set_npast(base + M);
        I.history.insert(I.history.end(), toks.begin(), toks.end());
    }
    return I.rows.data();
}

void Engine::prefill(const std::vector<token_t>& toks) {
    if (toks.empty()) return;
    Impl& I = *impl_;
    const double t0 = now_ms();

    std::vector<token_t> target = I.history;
    target.insert(target.end(), toks.begin(), toks.end());

    int64_t hit = 0;
    const PrefixCache::Snapshot* snap = I.prefix.find_longest(target, hit);
    if (snap) {
        const int64_t dk = I.kv.dk();
        for (int64_t l = 0; l < I.cfg.n_layer; ++l) {
            std::memcpy(I.kv.k(l), snap->k.data() + (size_t)(l * hit * dk),
                        (size_t)(hit * dk) * sizeof(float));
            std::memcpy(I.kv.v(l), snap->v.data() + (size_t)(l * hit * dk),
                        (size_t)(hit * dk) * sizeof(float));
        }
        I.kv.set_npast(hit);
        I.history.assign(target.begin(), target.begin() + hit);
    }

    size_t start = I.history.size() - (target.size() - toks.size());
    for (size_t i = hit; i < target.size();) {
        size_t j = std::min(target.size(), i + (size_t)I.opt.prefill_batch);
        ensure_capacity((int64_t)(j - i));
        forward(std::vector<token_t>(target.begin() + i, target.begin() + j),
                true, -1);
        i = j;
    }
    (void)start;

    if ((int64_t)target.size() >= I.prefix.block() * 2 &&
        (int64_t)target.size() <= I.ctx_cap) {
        const int64_t len = (int64_t)target.size();
        const int64_t dk = I.kv.dk();
        PrefixCache::Snapshot s;
        s.tokens = target;
        s.len = len;
        s.k.resize((size_t)(I.cfg.n_layer * len * dk));
        s.v.resize((size_t)(I.cfg.n_layer * len * dk));
        for (int64_t l = 0; l < I.cfg.n_layer; ++l) {
            std::memcpy(s.k.data() + (size_t)(l * len * dk), I.kv.k(l),
                        (size_t)(len * dk) * sizeof(float));
            std::memcpy(s.v.data() + (size_t)(l * len * dk), I.kv.v(l),
                        (size_t)(len * dk) * sizeof(float));
        }
        I.prefix.store(std::move(s));
    }

    impl_->st.prefill_tokens += toks.size();
    impl_->st.prefill_ms += now_ms() - t0;
}

const float* Engine::step(token_t t) {
    Impl& I = *impl_;
    const double t0 = now_ms();
    ensure_capacity(1);
    const float* out = forward({t}, true, -1);
    I.st.gen_ms += now_ms() - t0;
    I.st.gen_tokens += 1;
    return out;
}

std::vector<float> Engine::probe_rows(const std::vector<token_t>& toks,
                                      int layer_limit) {
    const float* r = forward(toks, false, layer_limit);
    const int64_t n = (int64_t)toks.size() * impl_->cfg.vocab;
    return std::vector<float>(r, r + n);
}

Engine::VerifyView Engine::extend_verify(
    const std::vector<token_t>& proposals) {
    token_t base_tok = history().empty() ? (token_t)-1 : history().back();
    const float* rows = forward(proposals, true, -1);
    return VerifyView{rows, (int64_t)proposals.size(), base_tok};
}

} // namespace ft
