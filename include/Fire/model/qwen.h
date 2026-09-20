#pragma once

#include "Fire/model/kv_cache.h"
#include "Fire/model/model.h"
#include "Fire/model/model_weights.h"
#include "Fire/op/add.h"
#include "Fire/op/embedding.h"
#include "Fire/op/linear.h"
#include "Fire/op/mha.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/op/rope.h"
#include "Fire/op/swiglu.h"

#include <memory>
#include <vector>

namespace model {

struct Qwen3Block {
    explicit Qwen3Block(float rms_norm_eps)
        : attention_norm(rms_norm_eps), q_norm(rms_norm_eps), k_norm(rms_norm_eps),
          ffn_norm(rms_norm_eps) {}

    op::RmsNormOp attention_norm;
    op::LinearOp wq;
    op::LinearOp wk;
    op::LinearOp wv;
    op::LinearOp wo;
    op::RmsNormOp q_norm;
    op::RmsNormOp k_norm;
    op::RmsNormOp ffn_norm;
    op::LinearOp w1;
    op::LinearOp w2;
    op::LinearOp w3;
};

struct Qwen3Runtime {
    tensor::Tensor token;        // INT32 [1]
    tensor::Tensor hidden;       // FP32 [hidden_size]
    tensor::Tensor block_output; // FP32 [hidden_size]
    tensor::Tensor norm_output;  // FP32 [hidden_size]

    // q_dim is explicit in Qwen3 and is not necessarily hidden_size.
    tensor::Tensor query; // FP32 [num_attention_heads, head_dim]
    tensor::Tensor key;   // FP32 [num_kv_heads, head_dim]
    tensor::Tensor value; // FP32 [num_kv_heads, head_dim]

    tensor::Tensor attention_score;     // FP32 [num_attention_heads, capacity]
    tensor::Tensor attention_output;    // FP32 [num_attention_heads, head_dim]
    tensor::Tensor attention_projected; // FP32 [hidden_size]
    tensor::Tensor attention_residual;  // FP32 [hidden_size]

    tensor::Tensor ffn_gate;      // FP32 [intermediate_size]
    tensor::Tensor ffn_up;        // FP32 [intermediate_size]
    tensor::Tensor ffn_activated; // FP32 [intermediate_size]
    tensor::Tensor ffn_down;      // FP32 [hidden_size]

    tensor::Tensor rope_cos; // FP32 [max_seq_len, head_dim / 2]
    tensor::Tensor rope_sin; // FP32 [max_seq_len, head_dim / 2]
    KVCache kv_cache;
};

// One implementation serves every dense Qwen3 profile. Profile selection is a
// data decision made by the loader, not a second model class or template type.
class Qwen3Model final : public Model {
  public:
    static base::Status create(const Qwen3Weights& weights, const op::OpContext& context,
                               std::unique_ptr<Qwen3Model>& output);

    ~Qwen3Model() override;

    const ModelConfig& config() const noexcept override;
    base::Status prepare(int32_t capacity, const op::OpContext& context) override;
    base::Status forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                         const op::OpContext& context) override;
    base::Status reset(const op::OpContext& context) override;

  private:
    explicit Qwen3Model(const Qwen3Weights& validated_weights);

    Qwen3Profile _profile;
    op::EmbeddingOp _embedding;
    std::vector<Qwen3Block> _layers;
    op::RoPEOp _rope;
    op::MultiHeadAttentionOp _mha;
    op::VecAddOp _add;
    op::RmsNormOp _norm;
    op::LinearOp _output;
    op::SwiGLUOp _swiglu;
    std::unique_ptr<Qwen3Runtime> _runtime;
};

} // namespace model
