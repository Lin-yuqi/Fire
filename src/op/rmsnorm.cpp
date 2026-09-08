#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "kernels/kernels_interface.h"
#include <Fire/op/rmsnorm.h>

// --------------------op begin-------------------
namespace op {

RmsNormOp::RmsNormOp() : ParamOperator(OpType::RMSNorm) {};
RmsNormOp::RmsNormOp(const float eps):ParamOperator(OpType::RMSNorm),_eps(eps){};
base::Status RmsNormOp::forward(const tensor::Tensor& input, tensor::Tensor& output,
                                OpContext context) {
    const int32_t dim = input.dims().empty() ? 0 : input.dims().back();
    auto status = _check(input, output, dim, context);
    if (!status)
        return status;


    // 这里如果要量化的话，可能还得做细分
    auto weight = _params[0]._data;
    auto dtype = context._device_type;
    if(input.dims().size()!=1){
        kernel::get_rmsnorm_kernel_dim(dtype)(input,weight,output,_eps,context._stream);
    }else{
        kernel::get_rmsnorm_kernel(dtype)(input,weight,output,_eps,context._stream);
    }

    return base::error::Success();
}

base::Status RmsNormOp::_check(const tensor::Tensor& input, tensor::Tensor& output, int32_t dim,
                               OpContext context) {
    if (dim <= 0) {
        return base::error::InvalidArgument("rmsnorm dimension must be positive");
    }
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("rmsnorm requires a CPU or GPU device");
    }
    if (_params.size() != 1) {
        return base::error::InvalidArgument("rmsnorm requires exactly one weight parameter");
    }
    if (_params[0].is_quantized()) {
        return base::error::InvalidArgument("rmsnorm does not support quantized weights");
    }
    if (device_type == base::DeviceType::CPU && input.dims().size() != 1) {
        return base::error::InvalidArgument("CPU rmsnorm only supports one-dimensional input");
    }

    auto status = _check_tensor(input, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(output, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    if (input.dims() != output.dims()) {
        return base::error::InvalidArgument("rmsnorm input and output shape mismatch");
    }
    return _check_tensor_with_dim(_params[0]._data, device_type, base::DataType::Fp32, {dim});
}

} // namespace op
// --------------------op end---------------------
