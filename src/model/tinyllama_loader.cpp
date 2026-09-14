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

base::Status TinyllamaLoader::load_weights(TinyLlamaWeights& output_weights) const {
    // TODO: check that the Reader is open, require TinyLlamaProfile::tensor_count
    // records, and validate every canonical name, FP32 dtype and expected shape.
    // TODO: after validating the entire profile, assemble CPU mmap views into a
    // local TinyLlamaWeights and publish it only when every field is complete.
    // No weight migration or Operator binding belongs in this entry point.
    (void)output_weights;
    return base::error::FunctionNotImplement(
        "TinyllamaLoader::load_weights: complete profile validation and assembly are not implemented");
}

} // namespace model
