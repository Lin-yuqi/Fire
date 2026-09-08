#pragma once
#include "Fire/tensor/tensor.h"
#include <Fire/base/base.h>
#include <Fire/op/operator.h>

// --------------------op begin-------------------
namespace op {

class RmsNormOp : public ParamOperator {
  public:
    explicit RmsNormOp();

    explicit RmsNormOp(const float eps);

    base::Status forward(const tensor::Tensor& input, tensor::Tensor& output,
                         OpContext context);

  private:
    base::Status _check(const tensor::Tensor& input, tensor::Tensor& output, int32_t dim,
                        OpContext context);

    float _eps = 1e-6;
};

} // namespace op
// --------------------op end---------------------