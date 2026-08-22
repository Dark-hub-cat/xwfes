#include "ft/gguf.h"

#include <cstdio>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace ft {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4,
                   T_I32 = 5, T_F32 = 6, T_BOOL = 7, T_STR = 8, T_ARR = 9,
                   T_U64 = 10, T_I64 = 11, T_F64 = 12;

struct Value {
    uint32_t type = 0;
    int64_t i = 0;
    double f = 0;
    bool b = false;
    std::string s;
    uint32_t arr_type = 0;
    uint64_t arr_count = 0;
    std::vector<std::string> arr_str;
    std::vector<uint8_t> arr_raw;
};

struct TensorInfo {
    std::string name;
    int64_t ne[4] = {1, 1, 1, 1};
    int32_t n_dims = 0;
    DType dtype = DType::F32;
    uint64_t offset = 0;
};

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}

    void need(size_t bytes) const {
        if (off_ + bytes > n_) throw std::runtime_error("gguf: truncated file");
    }

    template <typename T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + off_, sizeof(T));
        off_ += sizeof(T);
        return v;
    }

    std::string read_str() {
        uint64_t len = read<uint64_t>();
        need(len);
        std::string s((const char*)p_ + off_, len);
        off_ += len;
        return s;
    }

    Value read_value(uint32_t type) {
        Value v;
        v.type = type;
        switch (type) {
            case T_U8: v.i = read<uint8_t>(); break;
            case T_I8: v.i = read<int8_t>(); break;
            case T_U16: v.i = read<uint16_t>(); break;
            case T_I16: v.i = read<int16_t>(); break;
            case T_U32: v.i = read<uint32_t>(); break;
            case T_I32: v.i = read<int32_t>(); break;
            case T_F32: v.f = read<float>(); break;
            case T_BOOL: v.b = read<uint8_t>() != 0; break;
            case T_STR: v.s = read_str(); break;
            case T_U64: v.i = (int64_t)read<uint64_t>(); break;
            case T_I64: v.i = read<int64_t>(); break;
            case T_F64: v.f = read<double>(); break;
            case T_ARR: {
                v.arr_type = read<uint32_t>();
                v.arr_count = read<uint64_t>();
                if (v.arr_type == T_STR) {
                    v.arr_str.reserve(v.arr_count);
                    for (uint64_t i = 0; i < v.arr_count; ++i) v.arr_str.push_back(read_str());
                } else if (is_numeric(v.arr_type)) {
                    size_t esz = elem_size(v.arr_type);
                    v.arr_raw.resize(esz * v.arr_count);
                    need(v.arr_raw.size());
                    std::memcpy(v.arr_raw.data(), p_ + off_, v.arr_raw.size());
                    off_ += v.arr_raw.size();
                } else {
                    throw std::runtime_error("gguf: unsupported array element type");
                }
                break;
            }
            default:
                throw std::runtime_error("gguf: unsupported kv type");
        }
        return v;
    }

    size_t offset() const { return off_; }

private:
    static bool is_numeric(uint32_t t) {
        return t <= 7 || t == T_U64 || t == T_I64 || t == T_F64;
    }
    static size_t elem_size(uint32_t t) {
        switch (t) {
            case T_U8: case T_I8: case T_BOOL: return 1;
            case T_U16: case T_I16: return 2;
            case T_U32: case T_I32: case T_F32: return 4;
            case T_U64: case T_I64: case T_F64: return 8;
            default: return 0;
        }
    }
    const uint8_t* p_;
    size_t n_;
    size_t off_ = 0;
};

DType to_dtype(uint32_t t) {
    if (t == 0) return DType::F32;
    if (t == 1) return DType::F16;
    if (t == 2) return DType::Q4_0;
    if (t == 3) return DType::Q4_1;
    if (t == 8) return DType::Q8_0;
    if (t == 30) return DType::BF16;
    throw std::runtime_error("gguf: unsupported tensor dtype " + std::to_string(t));
}

} // namespace

struct Gguf::Impl {
#ifdef _WIN32
    HANDLE file = nullptr;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif
    const uint8_t* data = nullptr;
    size_t size = 0;
    std::unordered_map<std::string, Value> kv;
    std::unordered_map<std::string, TensorInfo> infos;
    uint64_t data_start = 0;

    void unmap() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file) CloseHandle(file);
        data = nullptr; mapping = nullptr; file = nullptr;
#else
        if (data && size) munmap((void*)data, size);
        if (fd >= 0) ::close(fd);
        data = nullptr; fd = -1;
#endif
    }
};

Gguf::Gguf(const std::string& path) : impl_(new Impl) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen > 0 ? wlen - 1 : 0, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER sz{};
    GetFileSizeEx(impl_->file, &sz);
    impl_->size = (size_t)sz.QuadPart;
    impl_->mapping = CreateFileMappingW(impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    impl_->data = (const uint8_t*)MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0);
#else
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st{};
    fstat(impl_->fd, &st);
    impl_->size = (size_t)st.st_size;
    impl_->data = (const uint8_t*)mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE,
                                       impl_->fd, 0);
#endif
    if (!impl_->data || impl_->size < 24) {
        impl_->unmap();
        throw std::runtime_error("gguf: empty or tiny file");
    }

    Reader r(impl_->data, impl_->size);
    if (r.read<uint32_t>() != GGUF_MAGIC) throw std::runtime_error("not a GGUF file");
    uint32_t version = r.read<uint32_t>();
    if (version != 2 && version != 3)
        throw std::runtime_error("unsupported GGUF version " + std::to_string(version));
    uint64_t n_tensors = r.read<uint64_t>();
    uint64_t n_kv = r.read<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.read_str();
        uint32_t t = r.read<uint32_t>();
        impl_->kv.emplace(std::move(key), r.read_value(t));
    }

    uint32_t align = 32;
    auto it = impl_->kv.find("general.alignment");
    if (it != impl_->kv.end()) align = (uint32_t)it->second.i;
    if (align == 0 || (align & (align - 1)) != 0) align = 32;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        ti.name = r.read_str();
        ti.n_dims = r.read<int32_t>();
        if (ti.n_dims < 1 || ti.n_dims > 4)
            throw std::runtime_error("gguf: bad dims for " + ti.name);
        for (int d = 0; d < 4; ++d) ti.ne[d] = 1;
        for (int d = 0; d < ti.n_dims; ++d) ti.ne[d] = (int64_t)r.read<uint64_t>();
        ti.dtype = to_dtype(r.read<uint32_t>());
        ti.offset = r.read<uint64_t>();
        impl_->infos.emplace(std::move(ti.name), std::move(ti));
    }

    uint64_t pos = r.offset();
    pos = (pos + align - 1) / align * align;
    if (pos > impl_->size) throw std::runtime_error("gguf: data out of bounds");
    impl_->data_start = pos;

    for (auto& [name, ti] : impl_->infos) {
        HostTensor ht;
        ht.name = name;
        for (int d = 0; d < 4; ++d) ht.ne[d] = ti.ne[d];
        ht.n_dims = ti.n_dims;
        ht.dtype = ti.dtype;
        ht.data = impl_->data + impl_->data_start + ti.offset;
        ht.nbytes = ht.slice_bytes() * (size_t)(ti.ne[2] * ti.ne[3]);
        if ((uint64_t)ht.nbytes > impl_->size - impl_->data_start - ti.offset)
            throw std::runtime_error("gguf: tensor out of bounds: " + name);
        tensors_.emplace(name, std::move(ht));
    }
}

Gguf::~Gguf() = default;

bool Gguf::has(const std::string& key) const {
    return impl_->kv.count(key) != 0;
}

std::string Gguf::get_str(const std::string& key, const std::string& def) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end()) return def;
    if (it->second.type == T_STR) return it->second.s;
    return def;
}

int64_t Gguf::get_i64(const std::string& key, int64_t def) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end()) return def;
    const Value& v = it->second;
    if (v.type <= T_F64 && v.type != T_STR) return v.i;
    return def;
}

double Gguf::get_f64(const std::string& key, double def) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end()) return def;
    const Value& v = it->second;
    if (v.type == T_F32 || v.type == T_F64) return v.f;
    if (v.type != T_STR) return (double)v.i;
    return def;
}

bool Gguf::get_bool(const std::string& key, bool def) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end()) return def;
    if (it->second.type == T_BOOL) return it->second.b;
    return def;
}

std::vector<std::string> Gguf::get_str_arr(const std::string& key) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end() || it->second.type != T_ARR ||
        it->second.arr_type != T_STR)
        return {};
    return it->second.arr_str;
}

static std::vector<float> raw_to_f32(const Value& v) {
    std::vector<float> out(v.arr_count);
    for (uint64_t i = 0; i < v.arr_count; ++i) {
        switch (v.arr_type) {
            case T_F32: {
                float f;
                std::memcpy(&f, v.arr_raw.data() + i * 4, 4);
                out[i] = f;
                break;
            }
            case T_F64: {
                double d;
                std::memcpy(&d, v.arr_raw.data() + i * 8, 8);
                out[i] = (float)d;
                break;
            }
            default: {
                int64_t x = 0;
                switch (v.arr_type) {
                    case T_I8: x = *(const int8_t*)(v.arr_raw.data() + i); break;
                    case T_U8: x = v.arr_raw[i]; break;
                    case T_I16: x = *(const int16_t*)(v.arr_raw.data() + i * 2); break;
                    case T_U16: x = *(const uint16_t*)(v.arr_raw.data() + i * 2); break;
                    case T_I32: x = *(const int32_t*)(v.arr_raw.data() + i * 4); break;
                    case T_U32: x = *(const uint32_t*)(v.arr_raw.data() + i * 4); break;
                    case T_I64: x = *(const int64_t*)(v.arr_raw.data() + i * 8); break;
                    case T_U64: x = (int64_t)*(const uint64_t*)(v.arr_raw.data() + i * 8); break;
                    default: x = 0;
                }
                out[i] = (float)x;
            }
        }
    }
    return out;
}

std::vector<float> Gguf::get_f32_arr(const std::string& key) const {
    auto it = impl_->kv.find(key);
    if (it == impl_->kv.end() || it->second.type != T_ARR) return {};
    return raw_to_f32(it->second);
}

std::vector<int32_t> Gguf::get_i32_arr(const std::string& key) const {
    auto f = get_f32_arr(key);
    std::vector<int32_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) out[i] = (int32_t)f[i];
    return out;
}

const HostTensor* Gguf::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

} // namespace ft
