#include <Fire/model/tinyllama_loader.h>

namespace model {

base::Status TinyllamaLoader::open(const std::string& path) { return _reader.open(path); }

base::Status TinyllamaLoader::loader_tensor(const std::string& name,
                                            tensor::Tensor& output_tensor) const {
    const TensorInfo* info = _reader.find(name);
    if (info == nullptr) {
        return base::error::ModelParseError("TinyLlama tensor not found: " + name);
    }

    auto buffer = _reader.mapped_buffer();

    output_tensor =
        tensor::Tensor(info->dtype, info->dims, buffer, static_cast<size_t>(info->byte_offset));
    return base::error::Success();
}

} // namespace model