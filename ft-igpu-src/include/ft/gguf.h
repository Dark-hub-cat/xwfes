#pragma once

#include "ft/types.h"

namespace ft {

class Gguf {
public:
    explicit Gguf(const std::string& path);
    ~Gguf();

    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;

    bool has(const std::string& key) const;

    std::string get_str(const std::string& key, const std::string& def = "") const;
    int64_t get_i64(const std::string& key, int64_t def = 0) const;
    double get_f64(const std::string& key, double def = 0) const;
    bool get_bool(const std::string& key, bool def = false) const;

    std::vector<std::string> get_str_arr(const std::string& key) const;
    std::vector<float> get_f32_arr(const std::string& key) const;
    std::vector<int32_t> get_i32_arr(const std::string& key) const;

    const HostTensor* tensor(const std::string& name) const;
    const std::unordered_map<std::string, HostTensor>& tensors() const { return tensors_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unordered_map<std::string, HostTensor> tensors_;
};

} // namespace ft
