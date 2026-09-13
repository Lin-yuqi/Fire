#include "Fire/op/linear.h"
#include "kernels/kernels_interface.h"

namespace op {

LinearOp::LinearOp() : ParamOperator(OpType::Linear) {}
LinearOp::LinearOp(float matmulop) : ParamOperator(OpType::Linear), _matmulop(matmulop) {}

base::Status LinearOp::forward(const tensor::Tensor& input, tensor::Tensor& output,
                               const OpContext& context) {
    auto status = _check(input,output,context);
    if(!status)return status;

    auto dtype = context._device_type;
    auto& weight = _params[0]._data;

    kernel::get_matmul_kernel(dtype)(input,weight,_matmulop,output,context._stream);
    if(_params.size()==2){
        auto& bias = _params[1]._data;
        kernel::get_add_kernel(dtype)(output,bias,output,context._stream);
    }
    
    return base::error::Success();
}

base::Status LinearOp::_check(const tensor::Tensor& input, tensor::Tensor& output,
                              const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("linear requires a CPU or GPU device");
    }

    if (_params.size() != 1 && _params.size() != 2) {
        return base::error::InvalidArgument(
            "linear requires one weight parameter and an optional bias parameter");
    }
    for (const auto& param : _params) {
        if (param.is_quantized()) {
            return base::error::InvalidArgument("linear does not support quantized parameters");
        }
    }

    auto status = _check_tensor(input, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(output, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }

    const auto& weight = _params[0]._data;
    status = _check_tensor(weight, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    if (weight.dims_size() != 2) {
        return base::error::InvalidArgument("linear weight must be two-dimensional");
    }
    if (input.dims_size() != 1 && input.dims_size() != 2) {
        return base::error::InvalidArgument("linear only supports one- or two-dimensional input");
    }
    if (output.dims_size() != input.dims_size()) {
        return base::error::InvalidArgument("linear input and output rank mismatch");
    }

    const int32_t input_features = weight.get_dim(1);
    const int32_t output_features = weight.get_dim(0);
    const int32_t last_dim = input.dims_size() - 1;
    if (input.get_dim(last_dim) != input_features) {
        return base::error::InvalidArgument("linear input feature dimension does not match weight");
    }
    if (output.get_dim(last_dim) != output_features) {
        return base::error::InvalidArgument(
            "linear output feature dimension does not match weight");
    }
    if (input.dims_size() == 2 && input.get_dim(0) != output.get_dim(0)) {
        return base::error::InvalidArgument("linear input and output row count mismatch");
    }

    if (_params.size() == 2) {
        status = _check_tensor_with_dim(_params[1]._data, device_type, base::DataType::Fp32,
                                        {output_features});
        if (!status) {
            return status;
        }
    }

    return base::error::Success();
}

} // namespace op
