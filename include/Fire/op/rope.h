#pragma once
#include <Fire/op/operator.h>
namespace op {

class RoPEOp : public Operator {
  public:
    base::Status forward(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                         const tensor::Tensor& sin, int32_t position, const OpContext& context);

  private:
    base::Status _check(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                        const tensor::Tensor& sin, int32_t position, const OpContext& context);
};

} // namespace op