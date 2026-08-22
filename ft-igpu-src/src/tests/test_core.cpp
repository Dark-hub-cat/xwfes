#include "ft/gguf.h"
#include "ft/graph.h"
#include "ft/quant.h"
#include "ft/chunker.h"
#include "ft/cpu_backend.h"
#include "ft/tokenizer.h"
#include "ft/engine.h"
#include "ft/spec.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

using namespace ft;

static int g_fails = 0;

#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %d: %s\n", __LINE__, msg);                  \
            ++g_fails;                                                    \
        }                                                                 \
    } while (0)

struct W {
    std::vector<uint8_t> b;
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) b.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) b.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
    }
    void i32(int32_t v) { u32((uint32_t)v); }
    void f32(float f) {
        uint32_t v;
        std::memcpy(&v, &f, 4);
        u32(v);
    }
    void str(const std::string& s) {
        u64(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
};

enum { T_U32 = 4, T_I32 = 5, T_F32 = 6, T_STR = 8, T_ARR = 9 };

static void kv_str(W& w, const std::string& k, const std::string& v) {
    w.str(k);
    w.u32(T_STR);
    w.str(v);
}
static void kv_u32(W& w, const std::string& k, uint32_t v) {
    w.str(k);
    w.u32(T_U32);
    w.u32(v);
}
static void kv_i32(W& w, const std::string& k, int32_t v) {
    w.str(k);
    w.u32(T_I32);
    w.i32(v);
}
static void kv_f32(W& w, const std::string& k, float v) {
    w.str(k);
    w.u32(T_F32);
    w.f32(v);
}
static void kv_arr_str(W& w, const std::string& k,
                       const std::vector<std::string>& v) {
    w.str(k);
    w.u32(T_ARR);
    w.u32(T_STR);
    w.u64(v.size());
    for (const auto& s : v) w.str(s);
}

static uint32_t g_lcg = 123456789u;
static float rnd() {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return ((float)((g_lcg >> 9) & 0x7FFF) / 32768.f) * 2.f - 1.f;
}

static const int64_t D = 16, H = 4, KVH = 2, HD = 4, L = 2, E = 4, FF = 32,
                     V = 27;

static bool write_model(const std::string& path) {
    W out;
    out.u32(0x46554747u);
    out.u32(3);

    size_t pos_tc = out.b.size();
    out.u64(0);
    size_t pos_kv = out.b.size();
    out.u64(0);

    uint32_t kvn = 0;
    auto cnt = [&] { ++kvn; };

    kv_str(out, "general.architecture", "qwen3moe");
    cnt();
    kv_u32(out, "qwen3moe.block_count", (uint32_t)L);
    cnt();
    kv_u32(out, "qwen3moe.embedding_length", (uint32_t)D);
    cnt();
    kv_u32(out, "qwen3moe.attention.head_count", (uint32_t)H);
    cnt();
    kv_u32(out, "qwen3moe.attention.head_count_kv", (uint32_t)KVH);
    cnt();
    kv_u32(out, "qwen3moe.attention.key_length", (uint32_t)HD);
    cnt();
    kv_u32(out, "qwen3moe.context_length", 128);
    cnt();
    kv_f32(out, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
    cnt();
    kv_f32(out, "qwen3moe.rope.freq_base", 10000.f);
    cnt();
    kv_str(out, "tokenizer.ggml.model", "llama");
    cnt();

    std::vector<std::string> toks;
    toks.push_back("<pad>");
    for (char c = 'a'; c <= 'z'; ++c) toks.push_back(std::string(1, c));
    kv_arr_str(out, "tokenizer.ggml.tokens", toks);
    cnt();
    kv_i32(out, "tokenizer.ggml.bos_token_id", -1);
    cnt();
    kv_i32(out, "tokenizer.ggml.eos_token_id", -1);
    cnt();

    struct TS {
        std::string name;
        int nd;
        int64_t ne[3];
        size_t bytes() const {
            size_t n = 1;
            for (int i = 0; i < nd; ++i) n *= (size_t)ne[i];
            return n * 4;
        }
    };
    std::vector<TS> ts;
    auto add = [&](std::string n, int nd, int64_t a, int64_t bb = 1,
                   int64_t cc = 1) {
        ts.push_back({n, nd, {a, bb, cc}});
    };

    add("token_embd.weight", 2, D, V);
    add("output_norm.weight", 1, D);
    add("output.weight", 2, D, V);
    for (int l = 0; l < (int)L; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", 1, D);
        add(p + "attn_q.weight", 2, D, H * HD);
        add(p + "attn_k.weight", 2, D, KVH * HD);
        add(p + "attn_v.weight", 2, D, KVH * HD);
        add(p + "attn_output.weight", 2, H * HD, D);
        add(p + "attn_q_norm.weight", 1, HD);
        add(p + "attn_k_norm.weight", 1, HD);
        add(p + "post_attention_norm.weight", 1, D);
        add(p + "ffn_gate_inp.weight", 2, D, E);
        add(p + "ffn_gate_exps.weight", 3, D, FF, E);
        add(p + "ffn_up_exps.weight", 3, D, FF, E);
        add(p + "ffn_down_exps.weight", 3, D, FF, E);
    }

    {
        const uint64_t tc = (uint64_t)ts.size();
        for (size_t i = 0; i < 8; ++i)
            out.b[pos_tc + i] = (uint8_t)((tc >> (8 * i)) & 0xFF);
    }
    {
        const uint64_t kv = (uint64_t)kvn;
        for (size_t i = 0; i < 8; ++i)
            out.b[pos_kv + i] = (uint8_t)((kv >> (8 * i)) & 0xFF);
    }

    uint64_t off = 0;
    for (auto& t : ts) {
        out.str(t.name);
        out.u32((uint32_t)t.nd);
        for (int d = 0; d < t.nd; ++d) out.u64((uint64_t)t.ne[d]);
        out.u32((uint32_t)DType::F32);
        out.u64(off);
        off += t.bytes();
        off = (off + 31) / 32 * 32;
    }

    uint64_t data_start = out.b.size();
    data_start = (data_start + 31) / 32 * 32;
    while ((uint64_t)out.b.size() < data_start) out.b.push_back(0);

    uint64_t cursor = 0;
    for (auto& t : ts) {
        bool ones = t.name.find("norm") != std::string::npos;
        size_t n = t.bytes() / 4;
        for (size_t i = 0; i < n; ++i) {
            float f = ones ? 1.0f : rnd() * 0.3f;
            out.f32(f);
        }
        cursor += t.bytes();
        uint64_t padto = (cursor + 31) / 32 * 32;
        while ((uint64_t)out.b.size() < data_start + padto)
            out.b.push_back(0);
        cursor = padto;
    }

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(out.b.data(), 1, out.b.size(), f);
    fclose(f);
    return true;
}

static void test_gguf(const std::string& path) {
    Gguf g(path);
    CHECK(g.get_str("general.architecture") == "qwen3moe", "arch str");
    ModelConfig c = config_from_gguf(g);
    CHECK(c.n_embd == D, "cfg embd");
    CHECK(c.n_layer == L, "cfg layers");
    CHECK(c.n_head == H && c.n_head_kv == KVH, "cfg heads");
    CHECK(c.head_dim == HD, "cfg hd");
    CHECK(c.vocab == V, "cfg vocab");
    const HostTensor* te = g.tensor("token_embd.weight");
    CHECK(te && te->rows() == V && te->cols() == D, "embd shape");
    const HostTensor* ge = g.tensor("blk.0.ffn_gate_exps.weight");
    CHECK(ge && ge->rows() == FF && ge->dim2() == E && ge->cols() == D,
          "exps shape");
}

static void test_tokenizer(Gguf& g) {
    Tokenizer tk = Tokenizer::from_gguf(g);
    auto ids = tk.encode("abc");
    CHECK(ids.size() == 3 && ids[0] == 1 && ids[1] == 2 && ids[2] == 3,
          "encode abc");
    CHECK(tk.decode(ids) == "abc", "decode roundtrip");
}

static void test_quant() {
    alignas(32) float ref[64];
    for (auto& v : ref) v = rnd();
    float x[64];
    for (auto& v : x) v = rnd();

    float refdot = 0;
    for (int i = 0; i < 64; ++i) refdot += ref[i] * x[i];

    std::vector<uint8_t> f16row(128);
    for (int i = 0; i < 64; ++i) {
        uint16_t h = float_to_half(ref[i]);
        std::memcpy(f16row.data() + i * 2, &h, 2);
    }
    float d16 = dot_row_f32(x, f16row.data(), DType::F16, 64);
    CHECK(std::fabs(d16 - refdot) / (std::fabs(refdot) + 1e-6f) < 0.02f,
          "f16 dot");

    std::vector<uint8_t> q8(64 / 32 * 34);
    for (int b = 0; b < 2; ++b) {
        float mx = 0;
        for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(ref[b * 32 + j]));
        float d = mx > 0 ? mx / 127.f : 1e-8f;
        uint16_t dh = float_to_half(d);
        std::memcpy(q8.data() + b * 34, &dh, 2);
        for (int j = 0; j < 32; ++j) {
            int qi = (int)std::lround(ref[b * 32 + j] / d);
            qi = std::clamp(qi, -127, 127);
            q8[(size_t)b * 34 + 2 + j] = (uint8_t)(int8_t)qi;
        }
    }
    float dq8 = dot_row_f32(x, q8.data(), DType::Q8_0, 64);
    CHECK(std::fabs(dq8 - refdot) / (std::fabs(refdot) + 1e-6f) < 0.02f,
          "q8_0 dot");

    std::vector<uint8_t> q4(64 / 32 * 18);
    for (int b = 0; b < 2; ++b) {
        float mx = 0;
        for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(ref[b * 32 + j]));
        float d = mx > 0 ? mx / 7.f : 1e-8f;
        uint16_t dh = float_to_half(d);
        std::memcpy(q4.data() + b * 18, &dh, 2);
        for (int j = 0; j < 32; ++j) {
            int nib = (int)std::lround(ref[b * 32 + j] / d) + 8;
            nib = std::clamp(nib, 0, 15);
            if (j & 1)
                q4[(size_t)b * 18 + 2 + (j >> 1)] |= (uint8_t)(nib << 4);
            else
                q4[(size_t)b * 18 + 2 + (j >> 1)] = (uint8_t)nib;
        }
    }
    float dq4 = dot_row_f32(x, q4.data(), DType::Q4_0, 64);
    CHECK(std::fabs(dq4 - refdot) / (std::fabs(refdot) + 1e-4f) < 0.35f,
          "q4_0 dot");
}

static HostTensor ht_view(std::vector<float>& v, int64_t cols, int64_t rows) {
    HostTensor t;
    t.name = "tmp";
    t.ne[0] = cols;
    t.ne[1] = rows;
    t.n_dims = 2;
    t.dtype = DType::F32;
    t.data = (const uint8_t*)v.data();
    t.nbytes = v.size() * 4;
    return t;
}

static void test_chunker() {
    const int64_t K = 16, FF = 48, M = 3, C = 8;
    std::vector<float> gate((size_t)(K * FF)), up((size_t)(K * FF)),
        down((size_t)(K * FF));
    for (auto& v : gate) v = rnd() * 0.5f;
    for (auto& v : up) v = rnd() * 0.5f;
    for (auto& v : down) v = rnd() * 0.5f;

    std::vector<float> x((size_t)(M * K));
    for (auto& v : x) v = rnd();

    CpuBackend cpu(1);
    StreamerConfig sc;
    sc.chunk_neurons = C;
    ExpertStreamer st(cpu, sc);
    std::vector<float> out((size_t)(M * K), 0.f);
    st.run(x.data(), M, K, ht_view(gate, K, FF), ht_view(up, K, FF),
           ht_view(down, K, FF), out.data());

    std::vector<float> ref((size_t)(M * K), 0.f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t i = 0; i < FF; ++i) {
            float g = dot_row_f32(&x[(size_t)(m * K)],
                                  (const uint8_t*)&gate[(size_t)(i * K)],
                                  DType::F32, K);
            float u = dot_row_f32(&x[(size_t)(m * K)],
                                  (const uint8_t*)&up[(size_t)(i * K)],
                                  DType::F32, K);
            float h = (g / (1.f + std::exp(-g))) * u;
            for (int64_t k = 0; k < K; ++k)
                ref[(size_t)(m * K + k)] += h * down[(size_t)(i * K + k)];
        }
    }
    double maxerr = 0;
    for (size_t j = 0; j < ref.size(); ++j)
        maxerr = std::max(maxerr,
                          (double)std::fabs(ref[j] - out[j]) /
                              (std::fabs(ref[j]) + 1e-3f));
    CHECK(maxerr < 1e-4, "chunked expert equivalence");
}

static void test_accept_chain() {
    const int64_t V = 4;
    float base[V] = {9, 1, 1, 1};
    float rows[2 * V] = {1, 9, 1, 1, 1, 1, 9, 1};
    token_t props1[3] = {0, 1, 2};
    token_t props2[3] = {0, 3, 2};
    CHECK(accept_chain_greedy(base, rows, V, props1, 3) == 3, "accept full");
    CHECK(accept_chain_greedy(base, rows, V, props2, 3) == 1, "reject early");
}

static void test_ngram() {
    NgramDrafter ng(4);
    std::vector<token_t> hist = {1, 2, 3, 1, 2, 3, 1, 2};
    ng.feed(hist);
    auto props = ng.draft({1, 2}, 4, nullptr);
    CHECK(!props.empty() && props[0] == 3, "ngram continuation");
    CHECK(ng.stats().hits >= 1, "ngram stats hit");
}

static void test_engine(const std::string& path) {
    EngineOptions eo;
    eo.model_path = path;
    eo.threads = 2;
    eo.igpu = IgpuMode::Off;
    eo.ctx_len = 128;
    eo.prefill_batch = 16;

    Engine eng;
    eng.load(eo);
    Tokenizer tk = eng.tok();
    auto ids = tk.encode("abcd");
    eng.prefill(ids);
    CHECK(eng.npast() == 4, "prefill npast");

    const float* lg = eng.step(5);
    CHECK(lg != nullptr, "step logits");
    CHECK(eng.history().back() == 5, "history append");

    auto probe = eng.probe_rows({1});
    CHECK(probe.size() == (size_t)eng.cfg().vocab, "probe rows size");

    eng.truncate_to(2);
    CHECK(eng.npast() == 2 && eng.history().size() == 2, "truncate");

    NgramDrafter ng(4);
    SampleParams sp;
    sp.temp = 0.f;
    GenResult gr =
        generate(eng, &ng, sp, 10, 4, [](token_t, const std::string&) {});
    CHECK(gr.produced > 0 && gr.produced <= 10, "generate produced range");
}

int main(int argc, char** argv) {
    const bool keep = argc > 1 && std::string(argv[1]) == "--keep";
    const std::string path = "ft_test_model.gguf";
    if (!write_model(path)) {
        std::printf("FAIL: cannot write test model\n");
        return 1;
    }
    try {
        test_gguf(path);
        Gguf g(path);
        test_tokenizer(g);
        test_quant();
        test_chunker();
        test_accept_chain();
        test_ngram();
        test_engine(path);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++g_fails;
    }
    if (!keep) std::remove(path.c_str());
    if (g_fails) {
        std::printf("%d checks FAILED\n", g_fails);
        return 1;
    }
    std::printf("all tests passed%s\n", keep ? " (model kept)" : "");
    return 0;
}
