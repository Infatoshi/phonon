#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

enum { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;
    template <typename T> T get() {
        if (sizeof(T) > (size_t) (end - p)) throw std::runtime_error("gguf: truncated");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (n > (size_t) (end - p)) throw std::runtime_error("gguf: truncated string");
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
    double scalar(uint32_t type) {
        switch (type) {
            case T_U8: return get<uint8_t>();
            case T_I8: return get<int8_t>();
            case T_U16: return get<uint16_t>();
            case T_I16: return get<int16_t>();
            case T_U32: return get<uint32_t>();
            case T_I32: return get<int32_t>();
            case T_F32: return get<float>();
            case T_BOOL: return get<uint8_t>();
            case T_U64: return (double) get<uint64_t>();
            case T_I64: return (double) get<int64_t>();
            case T_F64: return get<double>();
        }
        throw std::runtime_error("gguf: bad scalar type");
    }
};

// bytes per block and elements per block for the types this engine reads
void type_size(uint32_t type, uint64_t * bs, uint64_t * be) {
    switch (type) {
        case 0:  *bs = 4;   *be = 1;   return;  // F32
        case 1:  *bs = 2;   *be = 1;   return;  // F16
        case 8:  *bs = 34;  *be = 32;  return;  // Q8_0
        case 12: *bs = 144; *be = 256; return;  // Q4_K
        case 13: *bs = 176; *be = 256; return;  // Q5_K
        case 27: *bs = 8;   *be = 1;   return;  // I64
        case 14: *bs = 210; *be = 256; return;  // Q6_K
        default: *bs = 0;   *be = 1;   return;  // unsupported: nbytes 0
    }
}

}  // namespace

bool GGUF::open(const char * path, std::string * err) {
    try {
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error(std::string("cannot open ") + path);
        struct stat st;
        fstat(fd, &st);
        file_size = (size_t) st.st_size;
        const size_t page = (size_t) getpagesize();
        map_size = (file_size + page - 1) / page * page;
        map = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) { map = nullptr; throw std::runtime_error("mmap failed"); }

        Cursor c{ (const uint8_t *) map, (const uint8_t *) map + file_size };
        if (c.get<uint32_t>() != 0x46554747) throw std::runtime_error("not a GGUF file");
        if (c.get<uint32_t>() != 3) throw std::runtime_error("only GGUF v3");
        const uint64_t n_tensors = c.get<uint64_t>(), n_kv = c.get<uint64_t>();
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = c.str();
            GGUFValue v;
            const uint32_t type = c.get<uint32_t>();
            if (type == T_STR) {
                v.type = T_STR;
                v.str = c.str();
            } else if (type == T_ARR) {
                v.is_array = true;
                v.type = c.get<uint32_t>();
                const uint64_t n = c.get<uint64_t>();
                for (uint64_t j = 0; j < n; ++j) {
                    if (v.type == T_STR) c.str();  // string arrays (vocab) are skipped
                    else v.num.push_back(c.scalar(v.type));
                }
            } else {
                v.type = type;
                v.num.push_back(c.scalar(type));
            }
            kv[key] = std::move(v);
        }
        std::vector<GGUFTensor> infos;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            GGUFTensor t;
            t.name = c.str();
            const uint32_t nd = c.get<uint32_t>();
            uint64_t n_el = 1;
            for (uint32_t d = 0; d < nd; ++d) { t.ne.push_back(c.get<uint64_t>()); n_el *= t.ne.back(); }
            t.type = c.get<uint32_t>();
            t.offset = c.get<uint64_t>();
            uint64_t bs, be;
            type_size(t.type, &bs, &be);
            t.nbytes = n_el / be * bs;
            infos.push_back(t);
        }
        auto ait = kv.find("general.alignment");
        const uint64_t align = (ait != kv.end() && !ait->second.num.empty()) ? (uint64_t) ait->second.num[0] : 32;
        if (align == 0) throw std::runtime_error("gguf: zero alignment");
        const uint64_t pos = (uint64_t) (c.p - (const uint8_t *) map);
        const uint64_t data = (pos + align - 1) / align * align;
        for (auto & t : infos) {
            if (t.offset > file_size || data > file_size - t.offset || t.nbytes > file_size - (t.offset + data)) {
                throw std::runtime_error("gguf: tensor " + t.name + " out of bounds");
            }
            t.offset += data;
            tensors[t.name] = t;
        }
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

GGUF::~GGUF() {
    if (map) munmap(map, map_size);
}

const GGUFTensor & GGUF::t(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor " + name);
    return it->second;
}

double GGUF::num(const std::string & key) const {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.num.empty()) throw std::runtime_error("missing key " + key);
    return it->second.num[0];
}

std::vector<double> GGUF::arr(const std::string & key) const {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("missing key " + key);
    return it->second.num;
}
