#pragma once

#include "Fire/model/model.h"
#include "Fire/model/model_weights.h"
#include "Fire/op/add.h"
#include "Fire/op/linear.h"
#include "Fire/op/mha.h"
#include "Fire/op/operator.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/model/kv_cache.h"
#include "Fire/op/embedding.h"
#include "Fire/op/rope.h"
#include "Fire/op/swiglu.h"
namespace model {

// Parameter organization only. A default block has no bound weights and is
// not executable. Binding and block forward will follow the Loader work.
struct TinyLlamaBlock {
    op::RmsNormOp attention_norm{TinyLlamaProfile::rms_norm_eps};
    op::LinearOp wq;
    op::LinearOp wk;
    op::LinearOp wv;
    op::LinearOp wo;

    op::RmsNormOp ffn_norm{TinyLlamaProfile::rms_norm_eps};
    op::LinearOp w1;
    op::LinearOp w2;
    op::LinearOp w3;
};

struct TinyLlamaRuntime {
    // 模型输入与逐层传递的隐藏状态。
    tensor::Tensor token;        // INT32 [1]：当前 pos 位置的 token id，作为 Embedding 输入。
    tensor::Tensor hidden;       // FP32 [hidden_size]：Embedding 输出，也是当前 Block 的输入。
    tensor::Tensor block_output; // FP32 [hidden_size]：当前 Block 的最终输出，供下一层使用。
    tensor::Tensor norm_output;  // FP32 [hidden_size]：Attention/FFN/final RMSNorm 的复用输出。

    // 当前 Transformer Block 的 Q/K/V；第一维是 head，第二维是单个 head 的宽度。
    tensor::Tensor query; // FP32 [hidden_size] 投影后 reshape 为 [num_attention_heads, head_dim]。
    tensor::Tensor key;   // FP32 [kv_dim] 投影后 reshape 为 [num_kv_heads, head_dim] 并写入 cache。
    tensor::Tensor value; // FP32 [kv_dim] 投影后 reshape 为 [num_kv_heads, head_dim] 并写入 cache。

    // Attention 中间结果，只使用 attention_score 在 [0, pos] 范围内的有效前缀。
    tensor::Tensor attention_score;     // FP32 [num_attention_heads, capacity]：QK 注意力分数。
    tensor::Tensor attention_output;    // FP32 [num_attention_heads, head_dim]，Wo 前展平为 [hidden_size]。
    tensor::Tensor attention_projected; // FP32 [hidden_size]：Wo 对 attention_output 的投影。
    tensor::Tensor attention_residual;  // FP32 [hidden_size]：hidden 加 attention_projected。

    // FFN 中间结果；gate/up/activated 位于 intermediate_size 空间，down 回到 hidden_size。
    tensor::Tensor ffn_gate;      // FP32 [intermediate_size]：W1 gate 投影结果。
    tensor::Tensor ffn_up;        // FP32 [intermediate_size]：W3 up 投影结果。
    tensor::Tensor ffn_activated; // FP32 [intermediate_size]：SwiGLU(gate, up) 输出。
    tensor::Tensor ffn_down;      // FP32 [hidden_size]：W2 down 投影结果，等待残差相加。

    // RoPE 查找表；forward 使用第 pos 行，每个值对应一个前后半区维度对。
    tensor::Tensor rope_cos; // FP32 [max_seq_len, head_dim / 2]：各位置的 cos 值。
    tensor::Tensor rope_sin; // FP32 [max_seq_len, head_dim / 2]：各位置的 sin 值。

    // K/V 均为 FP32 [num_layers, capacity, num_kv_heads, head_dim]，pos 是序列位置维。
    KVCache kv_cache;
};

// TODO: implement TinyLlamaModel and its typed Runtime after structured weight
// loading, KVCache and the missing operators. Do not add device/stream state
// to this block or expose a generic KVCache requirement through Model.
class TinyLlamaModel final : public Model {
  public:
    // 验证完整权重结构并绑定参数。
    // 成功后才发布实例；失败时 output 保持原状。

    // creat实现模型的加载，包括总CPU->GPU的搬运
    static base::Status create(const TinyLlamaWeights& weights, op::OpContext context,
                               std::unique_ptr<TinyLlamaModel>& output);

    ~TinyLlamaModel() override;

    // 只读地暴露模型配置，比如层数、hidden size、head 数、KV head 数、vocab size
    // 等。它不应该修改任何模型状态。
    const ModelConfig& config() const noexcept override;

    /*负责“为推理运行准备资源”。核心是创建 _runtime，
    根据 capacity 申请 KV Cache 和各种中间 buffer。*/
    base::Status prepare(int32_t capacity, const op::OpContext& context) override;

    /*真正执行一次 token 的前向推理。一般流程是
    embedding → 逐层 Transformer block → final RMSNorm → LM Head → 输出 logits，
    同时把当前位置的 K/V 写入 KV Cache。*/
    base::Status forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                         const op::OpContext& context) override;

    /*reset(context)：清理“会话状态”，但不销毁模型。
    最典型就是把 KV Cache 的有效长度归零、把 runtime 中和上一轮生成相关的状态重置掉。
    它通常不需要重新申请显存，也不需要重新加载权重*/
    base::Status reset(const op::OpContext& context) override;

  private:
    explicit TinyLlamaModel(const TinyLlamaWeights& validated_weights);

    op::EmbeddingOp _embedding;
    std::vector<TinyLlamaBlock> _layers;
    op::RoPEOp _rope;
    op::MultiHeadAttentionOp _mha;
    op::VecAddOp _add;

    op::RmsNormOp _norm{TinyLlamaProfile::rms_norm_eps};

    op::LinearOp _output;

    op::SwiGLUOp _swiglu;

    // nullptr 表示尚未成功 prepare。
    std::unique_ptr<TinyLlamaRuntime> _runtime;
};

} // namespace model
