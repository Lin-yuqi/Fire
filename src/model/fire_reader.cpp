#include "Fire/model/fire_reader.h"
#include "Fire/base/base.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace model {

namespace {

constexpr size_t _HeaderSize = 32;
constexpr size_t _TensorInfoSize = 96;

constexpr uint32_t _FormatVersion = 1;
constexpr uint8_t _WireDTypeFp32 = 1;

constexpr char _Magic[] = "FIRECKPT";
constexpr size_t _HeaderMagicOffset = 0;
constexpr size_t _HeaderVersionOffset = 8;
constexpr size_t _HeaderTensorCountOffset = 12;
constexpr size_t _HeaderDirectoryOffset = 16;
constexpr size_t _HeaderDataOffset = 24;
constexpr size_t _TensorNameOffset = 0;
constexpr size_t _TensorByteOffsetOffset = 64;
constexpr size_t _TensorByteSizeOffset = 72;
constexpr size_t _TensorShape0Offset = 80;
constexpr size_t _TensorShape1Offset = 84;
constexpr size_t _TensorDTypeOffset = 88;
constexpr size_t _TensorNDimOffset = 89;
constexpr size_t _TensorPaddingOffset = 90;

bool is_supported_host() {
    const uint16_t value = 1;
    const auto* first_byte = reinterpret_cast<const uint8_t*>(&value);
    return sizeof(size_t) == sizeof(uint64_t) && sizeof(void*) == sizeof(uint64_t) &&
           first_byte[0] == 1;
}

uint32_t read_u32_le(const uint8_t* ptr) {
    return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* ptr) {
    return static_cast<uint64_t>(ptr[0]) | (static_cast<uint64_t>(ptr[1]) << 8) |
           (static_cast<uint64_t>(ptr[2]) << 16) | (static_cast<uint64_t>(ptr[3]) << 24) |
           (static_cast<uint64_t>(ptr[4]) << 32) | (static_cast<uint64_t>(ptr[5]) << 40) |
           (static_cast<uint64_t>(ptr[6]) << 48) | (static_cast<uint64_t>(ptr[7]) << 56);
}

class MMapBuffer final : public base::Buffer {
  public:
    MMapBuffer(void* ptr, size_t size)
        : base::Buffer(ptr, size, base::DeviceType::CPU), _mapping(ptr), _mapping_size(size) {}

    ~MMapBuffer() override {
        if (_mapping != nullptr && _mapping != MAP_FAILED) {
            ::munmap(_mapping, _mapping_size);
        }
    }

  private:
    void* _mapping = nullptr;
    size_t _mapping_size = 0;
};

class FileDescriptor final {
  public:
    explicit FileDescriptor(int fd) : _fd(fd) {}

    ~FileDescriptor() {
        if (_fd >= 0) {
            ::close(_fd);
        }
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    int get() const { return _fd; }

  private:
    int _fd = -1;
};

} // namespace

base::Status FireReader::open(const std::string& path) {
    if (path.empty())
        return base::error::PathNotValid("FireReader: path is empty");
    const int raw_fd = ::open(path.c_str(), O_RDONLY);
    if (raw_fd < 0)
        return base::error::PathNotValid("FireReader: failed to open file: " + path);

    FileDescriptor fd(raw_fd);

    if (!is_supported_host()) {
        return base::error::InternalError(
            "FireReader: Fire v1 requires a 64-bit little-endian host");
    }

    struct stat file_stat{};
    if (::fstat(fd.get(), &file_stat) != 0)
        return base::error::InternalError("FireReader: fstat failed");

    if (file_stat.st_size < static_cast<off_t>(_HeaderSize))
        return base::error::ModelParseError("FireReader: file is smaller than Fire v1 header");

    const size_t file_sz = static_cast<size_t>(file_stat.st_size);

    void* mapping = ::mmap(nullptr, file_sz, PROT_READ, MAP_PRIVATE, fd.get(), 0);

    if (mapping == MAP_FAILED)
        return base::error::InternalError("FireReader: mmap failed");

    auto local_buffer = std::make_shared<MMapBuffer>(mapping, file_sz);
    const auto* data = static_cast<const uint8_t*>(local_buffer->ptr());

    if (std::memcmp(data + _HeaderMagicOffset, _Magic, 8) != 0) {
        return base::error::ModelParseError("FireReader: invalid magic");
    }

    const uint32_t format_version = read_u32_le(data + _HeaderVersionOffset);

    if (format_version != _FormatVersion) {
        return base::error::ModelParseError("FireReader: unsupported format version");
    }

    const uint32_t tensor_count = read_u32_le(data + _HeaderTensorCountOffset);

    const uint64_t directory_offset = read_u64_le(data + _HeaderDirectoryOffset);

    if (directory_offset != _HeaderSize) {
        return base::error::ModelParseError("FireReader: invalid tensor directory offset");
    }

    const uint64_t data_offset = read_u64_le(data + _HeaderDataOffset);

    if (static_cast<uint64_t>(tensor_count) >
        (std::numeric_limits<uint64_t>::max() - _HeaderSize) / _TensorInfoSize) {
        return base::error::ModelParseError("FireReader: tensor directory size overflow");
    }

    const uint64_t expected_data_offset =
        static_cast<uint64_t>(_HeaderSize) +
        static_cast<uint64_t>(tensor_count) * static_cast<uint64_t>(_TensorInfoSize);

    if (data_offset != expected_data_offset) {
        return base::error::ModelParseError("FireReader: invalid data offset");
    }

    if (data_offset > file_sz) {
        return base::error::ModelParseError("FireReader: tensor directory exceeds file size");
    }

    std::vector<TensorInfo> local_tensors;
    std::unordered_map<std::string, size_t> local_index;

    uint64_t expected_payload_offset = data_offset;
    for (uint32_t i = 0; i < tensor_count; ++i) {
        const uint64_t entry_offset = directory_offset + static_cast<uint64_t>(i) * _TensorInfoSize;
        const auto* entry = data + static_cast<size_t>(entry_offset);

        // ---------------- name ----------------
        const uint8_t* name_ptr = entry + _TensorNameOffset;
        size_t name_length = 0;
        bool found_nul = false;

        for (size_t j = 0; j < 64; ++j) {
            const uint8_t ch = name_ptr[j];
            if (!found_nul) {
                if (ch == 0) {
                    found_nul = true;
                    name_length = j;
                } else if (ch > 0x7f) {
                    return base::error::ModelParseError(
                        "FireReader: tensor name contains non-ASCII byte");
                }
            } else {
                if (ch != 0) {
                    return base::error::ModelParseError(
                        "FireReader: non-zero byte after tensor name terminator");
                }
            }
        }

        if (!found_nul) {
            return base::error::ModelParseError("FireReader: tensor name is not NUL-terminated");
        }

        if (name_length == 0) {
            return base::error::ModelParseError("FireReader: tensor name is empty");
        }

        std::string name(reinterpret_cast<const char*>(name_ptr), name_length);

        if (local_index.find(name) != local_index.end()) {
            return base::error::ModelParseError("FireReader: duplicate tensor name: " + name);
        }

        // ---------------- offset / size ----------------
        const uint64_t byte_offset = read_u64_le(entry + _TensorByteOffsetOffset);

        const uint64_t byte_size = read_u64_le(entry + _TensorByteSizeOffset);

        // ---------------- shape ----------------
        const uint32_t shape0 = read_u32_le(entry + _TensorShape0Offset);

        const uint32_t shape1 = read_u32_le(entry + _TensorShape1Offset);

        // ---------------- dtype / ndim ----------------
        const uint8_t wire_dtype = *(entry + _TensorDTypeOffset);

        const uint8_t ndim = *(entry + _TensorNDimOffset);

        // ---------------- padding ----------------
        for (size_t j = 0; j < 6; ++j) {
            if (*(entry + _TensorPaddingOffset + j) != 0) {
                return base::error::ModelParseError("FireReader: tensor info padding must be zero");
            }
        }

        // ---------------- dtype validation ----------------
        base::DataType dtype = base::DataType::Unknown;

        if (wire_dtype == _WireDTypeFp32) {
            dtype = base::DataType::Fp32;
        } else {
            return base::error::ModelParseError("FireReader: unsupported tensor dtype");
        }

        // ---------------- ndim / shape validation ----------------
        if (ndim != 1 && ndim != 2) {
            return base::error::ModelParseError("FireReader: tensor ndim must be 1 or 2");
        }

        if (shape0 == 0) {
            return base::error::ModelParseError("FireReader: tensor shape contains zero dimension");
        }

        if (shape0 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            return base::error::ModelParseError("FireReader: tensor shape exceeds int32 range");
        }

        std::vector<int32_t> dims;

        uint64_t element_count = shape0;

        if (ndim == 1) {
            if (shape1 != 0) {
                return base::error::ModelParseError("FireReader: unused rank-1 shape must be zero");
            }

            dims.push_back(static_cast<int32_t>(shape0));
        } else {
            if (shape1 == 0) {
                return base::error::ModelParseError(
                    "FireReader: tensor shape contains zero dimension");
            }

            if (shape1 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
                return base::error::ModelParseError("FireReader: tensor shape exceeds int32 range");
            }

            if (static_cast<uint64_t>(shape0) >
                std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(shape1)) {
                return base::error::ModelParseError("FireReader: tensor element count overflow");
            }

            element_count = static_cast<uint64_t>(shape0) * static_cast<uint64_t>(shape1);

            dims.push_back(static_cast<int32_t>(shape0));
            dims.push_back(static_cast<int32_t>(shape1));
        }

        // ---------------- byte_size validation ----------------
        if (element_count > std::numeric_limits<uint64_t>::max() / 4) {
            return base::error::ModelParseError("FireReader: tensor byte size overflow");
        }

        const uint64_t expected_byte_size = element_count * 4;

        if (byte_size != expected_byte_size) {
            return base::error::ModelParseError(
                "FireReader: tensor byte_size does not match shape");
        }

        // Check the wire range before using offset ordering or calculating an
        // end position, so adversarial uint64 values cannot wrap.
        if (byte_size > std::numeric_limits<uint64_t>::max() - byte_offset) {
            return base::error::ModelParseError("FireReader: tensor payload range overflow");
        }

        // ---------------- payload alignment ----------------
        if (byte_offset % 4 != 0) {
            return base::error::ModelParseError("FireReader: tensor payload is not 4-byte aligned");
        }

        // ---------------- payload continuity ----------------
        if (byte_offset != expected_payload_offset) {
            return base::error::ModelParseError("FireReader: tensor payload is not tightly packed");
        }

        const uint64_t payload_end = byte_offset + byte_size;

        if (payload_end > file_sz) {
            return base::error::ModelParseError("FireReader: tensor payload exceeds file size");
        }

        expected_payload_offset = payload_end;

        // ---------------- decoded TensorInfo ----------------
        TensorInfo info;
        info.name = name;
        info.dtype = dtype;
        info.dims = std::move(dims);
        info.byte_offset = byte_offset;
        info.byte_size = byte_size;

        const size_t index = local_tensors.size();

        local_tensors.push_back(std::move(info));
        local_index.emplace(name, index);
    }
    if (expected_payload_offset != file_sz) {
        return base::error::ModelParseError(
            "FireReader: file size does not match final tensor payload");
    }
    _mapped_buffer = std::move(local_buffer);
    _tensors = std::move(local_tensors);
    _tensor_index = std::move(local_index);
    return base::error::Success();
}

const TensorInfo* FireReader::find(std::string_view name) const {
    const auto entry = _tensor_index.find(std::string(name));
    if (entry == _tensor_index.end()) {
        return nullptr;
    }
    return &_tensors.at(entry->second);
}

std::shared_ptr<base::Buffer> FireReader::mapped_buffer() const { return _mapped_buffer; }

} // namespace model
