#pragma once
#include "Fire/base/base.h"
#include "Fire/op/operator.h"

namespace op {
class EmbeddingOp : public ParamOperator {
  public:
    explicit EmbeddingOp();

    base::Status forward(const tensor::Tensor& tokens, tensor::Tensor& embeddings,
                         const OpContext& context) const;

  private:
    base::Status _check(const tensor::Tensor& tokens, tensor::Tensor& embeddings,
                        const OpContext& context) const;
};

} // namespace op