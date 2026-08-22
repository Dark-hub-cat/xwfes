#include "ft/tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace ft {

namespace {

void bytes_to_unicode(uint32_t out_cp[256]) {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                         (b >= 0xAE && b <= 0xFF);
        if (printable) out_cp[b] = (uint32_t)b;
        else out_cp[b] = 256 + (n++);
    }
}

uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = s[i++];
    if (c < 0x80) return c;
    uint32_t cp = 0;
    int extra = 0;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { return 0xFFFD; }
    while (extra-- && i < s.size()) {
        cp = (cp << 6) | ((unsigned char)s[i++] & 0x3F);
    }
    return cp;
}

void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
}

} // namespace

Tokenizer Tokenizer::from_gguf(const Gguf& g) {
    Tokenizer t;
    t.tokens_ = g.get_str_arr("tokenizer.ggml.tokens");
    if (t.tokens_.empty()) throw std::runtime_error("gguf: no tokenizer vocabulary");
    auto merges = g.get_str_arr("tokenizer.ggml.merges");
    std::string model = g.get_str("tokenizer.ggml.model",
                                  g.get_str("tokenizer.ggml.model_type", ""));
    t.gpt2_style_ = (model == "gpt2" || model == "bpe" || !merges.empty());

    for (int i = 0; i < (int)t.tokens_.size(); ++i) {
        t.id_of_.emplace(t.tokens_[i], (token_t)i);
    }

    t.bos_ = (token_t)g.get_i64("tokenizer.ggml.bos_token_id", -1);
    t.eos_ = (token_t)g.get_i64("tokenizer.ggml.eos_token_id", -1);

    static uint32_t b2c[256];
    static bool inited = false;
    if (!inited) { bytes_to_unicode(b2c); inited = true; }

    if (!t.gpt2_style_) {
        char buf[8];
        for (int b = 0; b < 256; ++b) {
            std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it = t.id_of_.find(buf);
            if (it != t.id_of_.end()) t.byte_fallback_[(uint32_t)b] = it->second;
        }
    } else {
        for (int b = 0; b < 256; ++b) {
            std::string piece;
            utf8_append(piece, b2c[b]);
            auto it = t.id_of_.find(piece);
            if (it != t.id_of_.end()) t.gpt2_byte_tok_[(uint8_t)b] = it->second;
        }
    }
    return t;
}

token_t Tokenizer::lookup_or_byte(const std::string& piece) const {
    auto it = id_of_.find(piece);
    if (it != id_of_.end()) return it->second;
    if (!piece.empty()) {
        uint8_t b = (uint8_t)piece[0];
        auto fb = byte_fallback_.find(b);
        if (fb != byte_fallback_.end()) return fb->second;
    }
    return -1;
}

std::vector<token_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<token_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);

    std::vector<std::string> syms;
    if (gpt2_style_) {
        static uint32_t b2c[256];
        static bool inited = false;
        if (!inited) { bytes_to_unicode(b2c); inited = true; }
        std::string mapped;
        for (char ch : text) {
            utf8_append(mapped, b2c[(uint8_t)ch]);
        }
        size_t i = 0;
        while (i < mapped.size()) {
            size_t best_len = 0;
            token_t best_id = -1;
            const size_t max_probe = std::min<size_t>(mapped.size() - i, 64);
            for (size_t l = max_probe; l >= 1; --l) {
                auto it = id_of_.find(mapped.substr(i, l));
                if (it != id_of_.end()) { best_len = l; best_id = it->second; break; }
            }
            if (!best_len) {
                uint32_t cp = utf8_next(mapped, i);
                std::string piece;
                utf8_append(piece, cp);
                syms.push_back(piece);
            } else {
                syms.push_back(mapped.substr(i, best_len));
                i += best_len;
            }
            (void)best_id;
        }
    } else {
        size_t i = 0;
        while (i < text.size()) {
            size_t best_len = 0;
            const size_t max_probe = std::min<size_t>(text.size() - i, 128);
            for (size_t l = max_probe; l >= 1; --l) {
                if (id_of_.count(text.substr(i, l))) { best_len = l; break; }
            }
            if (!best_len) {
                std::string piece = text.substr(i, 1);
                token_t id = lookup_or_byte(piece);
                if (id >= 0) { syms.push_back(piece); i += 1; continue; }
                i += 1;
                continue;
            }
            syms.push_back(text.substr(i, best_len));
            i += best_len;
        }
    }

    std::vector<int> word_start(syms.size(), 0);
    for (size_t j = 1; j < syms.size(); ++j) {
        if (syms[j].size() && (unsigned char)syms[j][0] < 0x80 &&
            !gpt2_style_ && syms[j][0] == ' ') {
            word_start[j] = 1;
        }
    }

    bool changed = true;
    std::vector<bool> alive(syms.size(), true);
    while (changed) {
        changed = false;
        token_t best_rank = INT32_MAX;
        size_t bi = (size_t)-1;
        std::string bm;
        for (size_t j = 0; j + 1 < syms.size(); ++j) {
            if (!alive[j] || !alive[j + 1]) continue;
            std::string merged = syms[j] + syms[j + 1];
            auto it = id_of_.find(merged);
            if (it == id_of_.end()) continue;
            token_t rid = it->second;
            if (rid < best_rank) {
                best_rank = rid;
                bi = j;
                bm = merged;
            }
        }
        if (bi != (size_t)-1) {
            syms[bi] = bm;
            alive[bi + 1] = false;
            changed = true;
        }
    }

    for (size_t j = 0; j < syms.size(); ++j) {
        if (!alive[j]) continue;
        token_t id = lookup_or_byte(syms[j]);
        if (id >= 0) out.push_back(id);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<token_t>& ids, bool skip_special) const {
    std::string raw;
    for (token_t id : ids) {
        if (id < 0 || id >= (int64_t)tokens_.size()) continue;
        if (skip_special && (id == bos_ || id == eos_)) continue;
        raw += tokens_[id];
    }
    std::string out;
    if (gpt2_style_) {
        static std::unordered_map<uint32_t, uint8_t> c2b;
        static bool inited = false;
        if (!inited) {
            uint32_t b2c[256];
            bytes_to_unicode(b2c);
            for (int b = 0; b < 256; ++b) c2b[b2c[b]] = (uint8_t)b;
            inited = true;
        }
        size_t i = 0;
        while (i < raw.size()) {
            uint32_t cp = utf8_next(raw, i);
            auto it = c2b.find(cp);
            if (it != c2b.end()) out += (char)it->second;
            else utf8_append(out, cp);
        }
    } else {
        size_t i = 0;
        while (i < raw.size()) {
            if (raw.compare(i, 1, "<") == 0 && i + 6 <= raw.size() &&
                raw[i + 1] == '0' && raw[i + 2] == 'x' && raw[i + 5] == '>') {
                auto hex = [](char c) -> int {
                    if (c >= '0' && c <= '9') return c - '0';
                    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                    return -1;
                };
                int hi = hex(raw[i + 3]), lo = hex(raw[i + 4]);
                if (hi >= 0 && lo >= 0) {
                    out += (char)((hi << 4) | lo);
                    i += 6;
                    continue;
                }
            }
            out += raw[i++];
        }
    }
    return out;
}

} // namespace ft
