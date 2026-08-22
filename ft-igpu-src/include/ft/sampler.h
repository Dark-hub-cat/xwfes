#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ft {

struct SampleParams {
    float temp = 0.7f;
    int top_k = 40;
    float top_p = 0.95f;
    uint64_t seed = 0;
};

class Sampler {
public:
    void reset(uint64_t seed) { state_ = seed ? seed : 0x9E3779B97F4A7C15ull; }

    static int argmax(const float* logits, int64_t n) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < (int)n; ++i) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        return best;
    }

    int sample(const float* logits, int64_t n, const SampleParams& p) {
        reset(p.seed ? p.seed : state_);
        if (p.temp <= 0.f) return argmax(logits, n);

        thread_local std::vector<float> probs;
        probs.assign((size_t)n, 0.f);
        float mx = logits[0];
        for (int i = 1; i < (int)n; ++i) mx = std::max(mx, logits[i]);
        double sum = 0;
        for (int i = 0; i < (int)n; ++i) {
            probs[i] = std::exp((logits[i] - mx) / p.temp);
            sum += probs[i];
        }
        for (int i = 0; i < (int)n; ++i) probs[i] = (float)(probs[i] / sum);

        if (p.top_k > 0 && p.top_k < (int)n) {
            std::vector<int> idx((size_t)n);
            for (int i = 0; i < (int)n; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + p.top_k, idx.end(),
                              [&](int a, int b) { return probs[a] > probs[b]; });
            float keep_sum = 0;
            for (int i = 0; i < p.top_k; ++i) keep_sum += probs[idx[i]];
            thread_local std::vector<float> filtered;
            filtered.assign((size_t)n, 0.f);
            for (int i = 0; i < p.top_k; ++i) filtered[idx[i]] = probs[idx[i]] / keep_sum;
            probs.swap(filtered);
        }

        if (p.top_p < 1.f) {
            std::vector<int> idx((size_t)n);
            for (int i = 0; i < (int)n; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });
            double cum = 0;
            size_t cut = idx.size();
            for (size_t i = 0; i < idx.size(); ++i) {
                cum += probs[idx[i]];
                if (cum >= p.top_p) { cut = i + 1; break; }
            }
            double tail = 0;
            for (size_t i = cut; i < idx.size(); ++i) { tail += probs[idx[i]]; probs[idx[i]] = 0; }
            if (tail > 0 && cut > 0) {
                for (size_t i = 0; i < cut; ++i)
                    probs[idx[i]] = (float)(probs[idx[i]] / (1.0 - tail));
            }
        }

        float r = next_float();
        double c = 0;
        for (int i = 0; i < (int)n; ++i) {
            c += probs[i];
            if ((float)c >= r) return i;
        }
        return argmax(logits, n);
    }

private:
    uint32_t next_u32() {
        state_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return (uint32_t)((z ^ (z >> 31)) >> 16);
    }
    float next_float() { return (float)(next_u32() >> 8) * (1.f / 16777216.f); }
    uint64_t state_ = 0x9E3779B97F4A7C15ull;
};

} // namespace ft
