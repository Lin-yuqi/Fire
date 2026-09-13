#include "Fire/op/matmul.h"
#include "Fire/base/base.h"
#include "kernels/kernels_interface.h"


namespace op {
MatmulOp::MatmulOp() : Operator(OpType::Matmul) {}

base::Status MatmulOp::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,float scale,
                               tensor::Tensor& output, const OpContext& context){
    auto status = _check(input1,input2,output,context);
    if(!status)return status;
    
    auto dtype = context._device_type;
    kernel::get_matmul_kernel(dtype)(input1,input2,scale,output,context._stream);
    return base::error::Success();
}

base::Status MatmulOp::_check(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              tensor::Tensor& output, const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("matmul requires a CPU or GPU device");
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

    if (input1.dims_size() != 1 && input1.dims_size() != 2) {
        return base::error::InvalidArgument("matmul only supports one- or two-dimensional input");
    }
    if (input2.dims_size() != 2) {
        return base::error::InvalidArgument("matmul weight must be two-dimensional");
    }
    if (output.dims_size() != input1.dims_size()) {
        return base::error::InvalidArgument("matmul input and output rank mismatch");
    }

    const int32_t input_features = input1.get_dim(input1.dims_size() - 1);
    const int32_t output_features = input2.get_dim(0);
    if (input_features != input2.get_dim(1)) {
        return base::error::InvalidArgument("matmul input feature dimension does not match weight");
    }
    if (output.get_dim(output.dims_size() - 1) != output_features) {
        return base::error::InvalidArgument(
            "matmul output feature dimension does not match weight");
    }
    if (input1.dims_size() == 2 && input1.get_dim(0) != output.get_dim(0)) {
        return base::error::InvalidArgument("matmul input and output row count mismatch");
    }

    return base::error::Success();
}

} // namespace op
