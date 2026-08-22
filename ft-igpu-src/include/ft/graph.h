#pragma once

#include "ft/gguf.h"

namespace ft {

struct ModelConfig {
    std::string arch;
    int64_t vocab = 0;
    int64_t n_embd = 0;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t n_ff = 0;
    int64_t n_experts = 0;
    int64_t n_expert_used = 0;
    bool moe = false;
    bool shared_expert = false;
    bool qk_norm = false;
    bool tied_embd = false;
    float rms_eps = 1e-5f;
    double rope_theta = 10000.0;
    bool rope_neox = true;
    int64_t ctx_train = 4096;

    std::string describe() const {
        std::string s = arch + ": embd=" + std::to_string(n_embd) +
                        " layers=" + std::to_string(n_layer) +
                        " heads=" + std::to_string(n_head) +
                        "/" + std::to_string(n_head_kv) +
                        " vocab=" + std::to_string(vocab);
        if (moe) {
            s += " MoE(" + std::to_string(n_experts) + "e x" +
                 std::to_string(n_expert_used) + ")";
        }
        return s;
    }
};

ModelConfig config_from_gguf(const Gguf& g);

class Weights {
public:
    explicit Weights(const Gguf& g) : g_(g) {}

    const HostTensor* t(const std::string& name) const { return g_.tensor(name); }

    const HostTensor* layer(int l, std::initializer_list<const char*> candidates) const {
        for (const char* c : candidates) {
            std::string name = fmt_layer(c, l);
            const HostTensor* t = g_.tensor(name);
            if (t) return t;
        }
        return nullptr;
    }

    static std::string fmt_layer(const char* pattern, int l) {
        std::string s(pattern);
        std::string num = std::to_string(l);
        size_t pos = 0;
        while ((pos = s.find("%d", pos)) != std::string::npos) {
            s.replace(pos, 2, num);
            pos += num.size();
        }
        return s;
    }

private:
    const Gguf& g_;
};

} // namespace ft
