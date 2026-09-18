#include <Fire/op/swiglu.h>
#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "kernels/kernels_interface.h"

namespace op {
SwiGLUOp::SwiGLUOp() : Operator(OpType::SwiGLU) {}

base::Status SwiGLUOp::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               tensor::Tensor& output, const OpContext& context) {
    auto status = _check(input1, input2, output, context);
    if (!status)
        return status;

    auto dtype = context._device_type;

    kernel::get_swiglu_kernel(dtype)(input1, input2, output, context._stream);

    return base::error::Success();
}

base::Status SwiGLUOp::_check(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              tensor::Tensor& output, const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("swiglu requires a CPU or GPU device");
    }

    auto status = _check_tensor(input1, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(input2, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(output, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }

    if (input1.dims() != input2.dims()) {
        return base::error::InvalidArgument("swiglu input shapes must match");
    }
    if (input1.dims() != output.dims()) {
        return base::error::InvalidArgument("swiglu input and output shapes must match");
    }

    return base::error::Success();
}

} // namespace op
