#include "Fire/base/base.h"
#include "kernels/kernels_interface.h"
#include <Fire/op/add.h>

// --------------op begin----------------
namespace op {
    VecAddOp::VecAddOp():Operator(OpType::Add){};

    base::Status VecAddOp::forward(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        tensor::Tensor& output,
        const OpContext& context
    ){
        auto status = _check(input1,input2,output,context);
        if(!status)return status;

        kernel::get_add_kernel(context._device_type)(input1,input2,output,context._stream);
        return base::error::Success();
    }

    base::Status VecAddOp::_check(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        tensor::Tensor& output,
        const OpContext& context
    ){
        int32_t size = input1.size();
        auto device_type=context._device_type;
        auto data_type=input1.data_type();
        auto status = _check_tensor(input1, device_type, data_type);
        if (!status) {
            return status;
        }
        status = _check_tensor(input2, device_type, data_type);
        if (!status) {
            return status;
        }
        status = _check_tensor(output, device_type, data_type);
        if (!status) {
            return status;
        }
        if (input1.dims() != input2.dims()) {
            return base::error::InvalidArgument(
                "input1 and input2 shape mismatch"
            );
        }
        if (input1.dims() != output.dims()) {
            return base::error::InvalidArgument(
                "input and output shape mismatch"
            );
        }
        return base::error::Success();
    }
}
// --------------op end------------------