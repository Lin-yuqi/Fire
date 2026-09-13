#pragma once
#include "Fire/tensor/tensor.h"
#include <Fire/base/base.h>
#include <Fire/op/operator.h>

namespace op {

class MatmulOp : public Operator {
  public:
    explicit MatmulOp();

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2, float scale,
                         tensor::Tensor& output, const OpContext& context);

  private:
    base::Status _check(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        tensor::Tensor& output, const OpContext& context);
};

} // namespace op