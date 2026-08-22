#include "ft/graph.h"

#include <algorithm>

namespace ft {

namespace {

int64_t kv_i(const Gguf& g, const std::string& arch,
             std::initializer_list<const char*> suffixes, int64_t def) {
    for (const char* s : suffixes) {
        int64_t v = g.get_i64(arch + "." + s, INT64_MIN);
        if (v != INT64_MIN) return v;
    }
    for (const char* s : suffixes) {
        int64_t v = g.get_i64(std::string("llama.") + s, INT64_MIN);
        if (v != INT64_MIN) return v;
    }
    return def;
}

} // namespace

ModelConfig config_from_gguf(const Gguf& g) {
    ModelConfig c;
    c.arch = g.get_str("general.architecture", "unknown");
    const std::string& a = c.arch;

    c.n_layer = kv_i(g, a, {"block_count"}, 0);
    c.n_embd = kv_i(g, a, {"embedding_length"}, 0);
    c.n_head = kv_i(g, a, {"attention.head_count"}, 0);
    c.n_head_kv = kv_i(g, a, {"attention.head_count_kv"}, c.n_head);
    c.head_dim = kv_i(g, a, {"attention.key_length",
                             "attention.head_dim"},
                      c.n_head ? c.n_embd / c.n_head : 0);
    c.vocab = kv_i(g, a, {"vocab_size"}, 0);
    if (!c.vocab) {
        auto toks = const_cast<Gguf&>(g).get_str_arr("tokenizer.ggml.tokens");
        c.vocab = (int64_t)toks.size();
    }
    c.ctx_train = kv_i(g, a, {"context_length"}, 4096);
    c.rms_eps = (float)kv_i(g, a, {"attention.layer_norm_rms_epsilon"}, 0);
    if (!c.rms_eps) c.rms_eps = 1e-5f;
    double theta = g.get_f64(a + ".rope.freq_base", 0);
    if (!theta) theta = g.get_f64("llama.rope.freq_base", 10000.0);
    c.rope_theta = theta;

    c.n_experts = kv_i(g, a, {"expert_count"}, 0);
    c.n_expert_used = kv_i(g, a, {"expert_used_count"}, 0);
    c.moe = c.n_experts > 1 && c.n_expert_used > 0;

    std::string al = a;
    std::transform(al.begin(), al.end(), al.begin(), ::tolower);
    c.rope_neox =
        al.find("qwen") != std::string::npos ||
        al.find("deepseek") != std::string::npos ||
        al.find("phi") != std::string::npos ||
        al.find("glm") != std::string::npos ||
        al.find("minicpm") != std::string::npos;

    if (!c.n_ff) {
        c.n_ff = kv_i(g, a, {"expert_feed_forward_length",
                             "feed_forward_length"}, 0);
    }

    if (!c.n_layer || !c.n_embd || !c.n_head || !c.head_dim || !c.vocab) {
        throw std::runtime_error("gguf: incomplete model config for arch " + a);
    }
    return c;
}

const HostTensor* find_tensor_any(const Gguf& g, const std::vector<std::string>& names) {
    for (const auto& n : names) {
        const HostTensor* t = g.tensor(n);
        if (t) return t;
    }
    return nullptr;
}

} // namespace ft
