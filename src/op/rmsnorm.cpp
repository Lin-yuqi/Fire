#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "kernels/kernels_interface.h"
#include <Fire/op/rmsnorm.h>

// --------------------op begin-------------------
namespace op {

RmsNormOp::RmsNormOp() : ParamOperator(OpType::RMSNorm) {};

base::Status RmsNormOp::forward(const tensor::Tensor& input, tensor::Tensor& output, int32_t dim,
                                OpContext context) {
    auto status = _check(input, output, dim, context);
    if (!status)
        return status;

    // 这里如果要量化的话，可能还得做细分
    auto weight = _params[0]._data;
    kernel::get_rmsnorm_kernel(context._device_type)(input, weight, output,eps,context._stream);
    return base::error::Success();
}

base::Status RmsNormOp::_check(const tensor::Tensor& input, tensor::Tensor& output, int32_t dim,
                                OpContext context) {

}

} // namespace op
// --------------------op end---------------------