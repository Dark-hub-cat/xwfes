#pragma once

#include "ft/gguf.h"

#include <memory>
#include <string>
#include <vector>

namespace ft {

class Tokenizer {
public:
    static Tokenizer from_gguf(const Gguf& g);

    std::vector<token_t> encode(const std::string& text, bool add_bos = false) const;
    std::string decode(const std::vector<token_t>& ids, bool skip_special = true) const;

    token_t bos() const { return bos_; }
    token_t eos() const { return eos_; }
    int64_t vocab_size() const { return (int64_t)tokens_.size(); }
    const std::vector<std::string>& tokens() const { return tokens_; }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, token_t> id_of_;
    std::unordered_map<uint32_t, token_t> byte_fallback_;
    std::unordered_map<uint8_t, token_t> gpt2_byte_tok_;
    bool gpt2_style_ = false;
    token_t bos_ = -1, eos_ = -1;

    token_t lookup_or_byte(const std::string& piece) const;
};

} // namespace ft
