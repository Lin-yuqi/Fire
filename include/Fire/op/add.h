#pragma once
#include "Fire/tensor/tensor.h"
#include <Fire/base/base.h>
#include <Fire/op/operator.h>

// -------------------op begin------------------
namespace op {

class VecAddOp : public Operator{
public:
    explicit VecAddOp();

    base::Status forward(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        tensor::Tensor& output,
        const OpContext& context
    );
private:
    base::Status _check(
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        tensor::Tensor& output,
        const OpContext& context
    );

};


}
// -------------------op end-------------------- 