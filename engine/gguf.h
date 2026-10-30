// Minimal GGUF v3 reader: memory-maps the file, parses metadata and tensor infos.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct GGUFTensor {
    std::string name;
    uint32_t type;           // ggml type id (0 F32, 12 Q4_K, 14 Q6_K, ...)
    std::vector<uint64_t> ne;
    uint64_t offset;         // absolute byte offset in the file
    uint64_t nbytes;
};

struct GGUFValue {
    uint32_t type = 0;       // gguf value type of the scalar or of the array elements
    bool is_array = false;
    std::vector<double> num;  // numeric scalars / arrays (bools as 0/1)
    std::string str;
};

struct GGUF {
    void * map = nullptr;
    size_t map_size = 0;     // page-rounded mapping length
    size_t file_size = 0;
    std::map<std::string, GGUFValue> kv;
    std::map<std::string, GGUFTensor> tensors;

    bool open(const char * path, std::string * err);
    ~GGUF();
    const GGUFTensor & t(const std::string & name) const;
    bool has(const std::string & name) const { return tensors.count(name) != 0; }
    double num(const std::string & key) const;
    std::vector<double> arr(const std::string & key) const;
};
