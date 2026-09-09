#include "Fire/model/fire_reader.h"

namespace model {

base::Status FireReader::open(const std::string& path) {
    (void)path;
    return base::error::FunctionNotImplement(
        "FireReader::open scaffold is present; wire parsing is not implemented yet");
}

const TensorInfo* FireReader::find(std::string_view name) const {
    const auto entry = tensor_index_.find(std::string(name));
    if (entry == tensor_index_.end()) {
        return nullptr;
    }
    return &tensors_.at(entry->second);
}

std::shared_ptr<base::Buffer> FireReader::mapped_buffer() const {
    return mapped_buffer_;
}

}  // namespace model
