#pragma once
#include <Fire/op/operator.h>
namespace op {
class SwiGLUOp : public Operator {
  public:
    explicit SwiGLUOp();

    base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                         tensor::Tensor& output, OpContext& context);

  private:
    base::Status _check(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        tensor::Tensor& output, OpContext& context);
};
} // namespace op