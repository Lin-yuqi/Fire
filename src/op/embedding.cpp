#include "Fire/op/embedding.h"
#include "Fire/base/base.h"
#include "kernels/kernels_interface.h"

namespace op {
EmbeddingOp::EmbeddingOp() : ParamOperator(OpType::Embedding) {};

base::Status EmbeddingOp::forward(const tensor::Tensor& tokens, tensor::Tensor& embeddings,
                                  const OpContext& context) const {
    auto status = _check(tokens, embeddings, context);
    if (!status)
        return status;
    auto wei = _params[0]._data;

    auto dtype = context._device_type;
    kernel::get_embedding_kernel(dtype)(tokens, wei, embeddings, context._stream);
    return base::error::Success();
}

base::Status EmbeddingOp::_check(const tensor::Tensor& tokens, tensor::Tensor& embeddings,
                                 const OpContext& context) const {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("embedding requires a CPU or GPU device");
    }

    if (_params.size() != 1) {
        return base::error::InvalidArgument("embedding requires exactly one weight parameter");
    }
    if (_params[0].is_quantized()) {
        return base::error::InvalidArgument("embedding does not support quantized weights");
    }

    auto status = _check_tensor(tokens, device_type, base::DataType::int32);
    if (!status) {
        return status;
    }
    status = _check_tensor(embeddings, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }

    const auto& weight = _params[0]._data;
    status = _check_tensor(weight, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    if (weight.dims_size() != 2) {
        return base::error::InvalidArgument("embedding weight must be two-dimensional");
    }

    const auto embedding_dim = static_cast<size_t>(weight.get_dim(1));
    if (embeddings.size() != tokens.size() * embedding_dim) {
        return base::error::InvalidArgument(
            "embedding output size does not match token count and weight dimension");
    }

    return base::error::Success();
}

} // namespace op
