#pragma once
#include <Fire/op/operator.h>

namespace op {
class MultiHeadAttentionOp : public Operator {

  public:
    explicit MultiHeadAttentionOp();
    // q:[num_q_heads,head_dim]
    // key_cache、val_cache:[num_layers, capacity, num_kv_heads, head_dim]
    // score:[TinyLlamaProfile::num_attention_heads, capacity]
    // output:[num_attention_heads, head_dim]
    base::Status forward(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                         const tensor::Tensor& val_cache, tensor::Tensor& score,
                         tensor::Tensor& output, int32_t layer_idx, int32_t pos,
                         const OpContext& context);

  private:
    base::Status _check(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                        const tensor::Tensor& val_cache, tensor::Tensor& score,
                        tensor::Tensor& output, int32_t layer_idx, int32_t pos,
                        const OpContext& context);
};
} // namespace op
