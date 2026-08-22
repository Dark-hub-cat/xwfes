#pragma once

#include "ft/types.h"

#include <algorithm>
#include <cstring>

namespace ft {

class KvCache {
public:
    void alloc(int64_t n_layers, int64_t capacity, int64_t dim_k, int64_t dim_v) {
        n_layers_ = n_layers;
        cap_ = capacity;
        dk_ = dim_k;
        dv_ = dim_v;
        k_.assign((size_t)(n_layers * capacity * dim_k), 0.f);
        v_.assign((size_t)(n_layers * capacity * dim_v), 0.f);
        npast_ = 0;
    }

    float* k(int64_t layer) { return k_.data() + (size_t)(layer * cap_ * dk_); }
    float* v(int64_t layer) { return v_.data() + (size_t)(layer * cap_ * dv_); }
    const float* k(int64_t layer) const { return k_.data() + (size_t)(layer * cap_ * dk_); }
    const float* v(int64_t layer) const { return v_.data() + (size_t)(layer * cap_ * dv_); }

    int64_t dk() const { return dk_; }
    int64_t dv() const { return dv_; }
    int64_t cap() const { return cap_; }
    int64_t npast() const { return npast_; }
    void set_npast(int64_t p) { npast_ = p; }

    void clear() { npast_ = 0; }

    void copy_prefix_from(const KvCache& src, int64_t len) {
        const int64_t layers = std::min(n_layers_, src.n_layers_);
        for (int64_t l = 0; l < layers; ++l) {
            std::memcpy(k(l), src.k(l), (size_t)(len * dk_) * sizeof(float));
            std::memcpy(v(l), src.v(l), (size_t)(len * dv_) * sizeof(float));
        }
        npast_ = len;
    }

private:
    int64_t n_layers_ = 0, cap_ = 0, dk_ = 0, dv_ = 0, npast_ = 0;
    std::vector<float> k_, v_;
};

class PrefixCache {
public:
    struct Snapshot {
        std::vector<token_t> tokens;
        std::vector<float> k, v;
        int64_t len = 0;
        uint64_t touched = 0;
    };

    explicit PrefixCache(int64_t block = 32, size_t max_entries = 8)
        : block_(block), max_entries_(max_entries), tick_(0) {}

    const Snapshot* find_longest(const std::vector<token_t>& tokens,
                                 int64_t& out_len) const {
        const Snapshot* best = nullptr;
        int64_t best_len = 0;
        for (const auto& s : entries_) {
            if ((int64_t)s.tokens.size() > (int64_t)tokens.size()) continue;
            int64_t match = prefix_match(s.tokens, tokens);
            match -= match % block_;
            if (match > best_len && match >= block_) {
                best_len = match;
                best = &s;
            }
        }
        out_len = best_len;
        return best_len >= block_ ? best : nullptr;
    }

    void store(Snapshot&& s) {
        if ((int64_t)s.tokens.size() < block_) return;
        s.touched = ++tick_;
        entries_.push_back(std::move(s));
        if (entries_.size() > max_entries_) {
            auto oldest = std::min_element(entries_.begin(), entries_.end(),
                [](const Snapshot& a, const Snapshot& b) { return a.touched < b.touched; });
            entries_.erase(oldest);
        }
    }

    int64_t block() const { return block_; }

private:
    static int64_t prefix_match(const std::vector<token_t>& a,
                                const std::vector<token_t>& b) {
        int64_t i = 0;
        while (i < (int64_t)a.size() && i < (int64_t)b.size() && a[i] == b[i]) ++i;
        return i;
    }

    int64_t block_;
    size_t max_entries_;
    mutable uint64_t tick_;
    std::vector<Snapshot> entries_;
};

} // namespace ft
