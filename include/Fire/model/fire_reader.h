#pragma once

#include "Fire/base/base.h"
#include "Fire/base/buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace model {

// Decoded tensor metadata. These fields describe Fire tensor semantics rather
// than the packed 96-byte wire record.
struct TensorInfo {
    std::string name;
    base::DataType dtype = base::DataType::Unknown;
    std::vector<int32_t> dims;
    uint64_t byte_offset = 0;
    uint64_t byte_size = 0;
};

// Owns and indexes one read-only mmap of a .fire v1 tensor container.
//
// FireReader only understands the container format. Model-specific tensor
// requirements belong to ModelLoader.
class FireReader final {
public:
    FireReader() = default;
    ~FireReader() = default;

    FireReader(const FireReader&) = delete;
    FireReader& operator=(const FireReader&) = delete;
    FireReader(FireReader&&) = delete;
    FireReader& operator=(FireReader&&) = delete;

    // Opens and validates a complete .fire file. Malformed external input must
    // be reported as Status and must never reach CHECK/LOG(FATAL).
    base::Status open(const std::string& path);

    // Returns Reader-owned metadata. The pointer is valid only while this
    // Reader remains alive. An unopened Reader and a missing name both return
    // nullptr.
    const TensorInfo* find(std::string_view name) const;

    // Shares ownership of the whole mmap with CPU Tensor views. Returns an
    // empty shared_ptr before a successful open.
    std::shared_ptr<base::Buffer> mapped_buffer() const;

private:
    std::shared_ptr<base::Buffer> _mapped_buffer;
    std::vector<TensorInfo> _tensors;
    std::unordered_map<std::string, size_t> _tensor_index;
};

}  // namespace model
