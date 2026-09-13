#pragma once
#include "Fire/tensor/tensor.h"
#include <Fire/base/base.h>
#include <Fire/op/operator.h>

namespace op {
class LinearOp : public ParamOperator {
  public:
    explicit LinearOp();
    explicit LinearOp(float matmulop);

    base::Status forward(const tensor::Tensor& input, tensor::Tensor& output,
                         const OpContext& context);

  private:
    base::Status _check(const tensor::Tensor& input, tensor::Tensor& output,
                        const OpContext& context);

    float _matmulop = 1.0f;
};

} // namespace op
