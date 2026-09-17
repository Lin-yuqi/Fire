#include <Fire/op/mha.h>

#include <vector>

#include "Fire/base/base.h"
#include "kernels/kernels_interface.h"

namespace op {
MultiHeadAttentionOp::MultiHeadAttentionOp() : Operator(OpType::MHA) {}

base::Status MultiHeadAttentionOp::forward(const tensor::Tensor& input_q,
                                           const tensor::Tensor& key_cache,
                                           const tensor::Tensor& val_cache, tensor::Tensor& score,
                                           tensor::Tensor& output, int32_t layer_idx, int32_t pos,
                                           const OpContext& context) {
    auto status =
        _check(input_q, key_cache, val_cache, score, output, layer_idx, pos, context);
    if (!status) {
        return status;
    }

    kernel::get_mha_kernel(context._device_type)(input_q, key_cache, val_cache, score, output,
                                                  layer_idx, pos, context._stream);
    return base::error::Success();
}

base::Status MultiHeadAttentionOp::_check(const tensor::Tensor& input_q,
                                          const tensor::Tensor& key_cache,
                                          const tensor::Tensor& val_cache, tensor::Tensor& score,
                                          tensor::Tensor& output, int32_t layer_idx, int32_t pos,
                                          const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("mha requires a CPU or GPU device");
    }

    const tensor::Tensor* tensors[] = {&input_q, &key_cache, &val_cache, &score, &output};
    for (const auto* tensor : tensors) {
        auto status = _check_tensor(*tensor, device_type, base::DataType::Fp32);
        if (!status) {
            return status;
        }
    }

    if (input_q.dims_size() != 2) {
        return base::error::InvalidArgument("mha query must be two-dimensional");
    }
    if (key_cache.dims_size() != 4 || val_cache.dims_size() != 4) {
        return base::error::InvalidArgument("mha key and value caches must be four-dimensional");
    }
    if (score.dims_size() != 2 || output.dims_size() != 2) {
        return base::error::InvalidArgument("mha score and output must be two-dimensional");
    }
    if (key_cache.dims() != val_cache.dims()) {
        return base::error::InvalidArgument("mha key and value cache shapes must match");
    }

    const int32_t num_q_heads = input_q.get_dim(0);
    const int32_t head_dim = input_q.get_dim(1);
    const int32_t num_layers = key_cache.get_dim(0);
    const int32_t capacity = key_cache.get_dim(1);
    const int32_t num_kv_heads = key_cache.get_dim(2);
    const int32_t kv_head_dim = key_cache.get_dim(3);

    if (layer_idx < 0 || layer_idx >= num_layers) {
        return base::error::InvalidArgument("mha layer index is outside the cache");
    }
    if (pos < 0 || pos >= capacity) {
        return base::error::InvalidArgument("mha position is outside the cache capacity");
    }
    if (head_dim != kv_head_dim) {
        return base::error::InvalidArgument("mha query and cache head dimensions must match");
    }
    if (num_kv_heads > num_q_heads || num_q_heads % num_kv_heads != 0) {
        return base::error::InvalidArgument(
            "mha query head count must be divisible by the KV head count");
    }
    if (score.dims() != std::vector<int32_t>{num_q_heads, capacity}) {
        return base::error::InvalidArgument("mha score shape must be [query heads, capacity]");
    }
    if (output.dims() != std::vector<int32_t>{num_q_heads, head_dim}) {
        return base::error::InvalidArgument("mha output shape must match the query shape");
    }

    return base::error::Success();
}

} // namespace op
