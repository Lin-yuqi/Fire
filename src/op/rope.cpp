#include <Fire/op/rope.h>
#include "kernels/kernels_interface.h"
namespace op {
RoPEOp::RoPEOp() : Operator(OpType::RoPE) {}

base::Status RoPEOp::forward(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                             const tensor::Tensor& sin, int32_t position,
                             const OpContext& context) {
    auto status = _check(query, key, cos, sin, position, context);
    if (!status)
        return status;

    auto dtype = context._device_type;

    kernel::get_rope_kernel(dtype)(query, key, cos, sin, position, context._stream);
    return base::error::Success();
}

base::Status RoPEOp::_check(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                            const tensor::Tensor& sin, int32_t position, const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("rope requires a CPU or GPU device");
    }
    if (position < 0) {
        return base::error::InvalidArgument("rope position must be non-negative");
    }

    auto status = _check_tensor(query, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(key, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(cos, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(sin, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }

    if (query.dims_size() != 2 || key.dims_size() != 2) {
        return base::error::InvalidArgument("rope query and key must be two-dimensional");
    }
    if (cos.dims_size() != 2 || sin.dims_size() != 2) {
        return base::error::InvalidArgument("rope cos and sin caches must be two-dimensional");
    }

    const int32_t head_dim = query.get_dim(1);
    if (head_dim <= 0 || head_dim % 2 != 0) {
        return base::error::InvalidArgument("rope head dimension must be positive and even");
    }
    if (key.get_dim(1) != head_dim) {
        return base::error::InvalidArgument("rope query and key head dimensions must match");
    }
    if (key.get_dim(0) > query.get_dim(0)) {
        return base::error::InvalidArgument("rope key head count must not exceed query head count");
    }

    if (cos.dims() != sin.dims()) {
        return base::error::InvalidArgument("rope cos and sin cache shapes must match");
    }
    if (cos.get_dim(1) != head_dim / 2) {
        return base::error::InvalidArgument(
            "rope cache width must be half the query and key head dimension");
    }
    if (position >= cos.get_dim(0)) {
        return base::error::InvalidArgument("rope position exceeds the cache length");
    }

    return base::error::Success();
}

} // namespace op
