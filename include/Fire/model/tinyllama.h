#pragma once

#include "Fire/model/model.h"
#include "Fire/model/model_weights.h"
#include "Fire/op/linear.h"
#include "Fire/op/operator.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/model/kv_cache.h"
#include "Fire/op/embedding.h"
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
    tensor::Tensor hidden;
    tensor::Tensor block_output;
    tensor::Tensor norm_output;

    tensor::Tensor query;
    tensor::Tensor key;
    tensor::Tensor value;

    tensor::Tensor attention_score;
    tensor::Tensor attention_output;
    tensor::Tensor attention_projected;
    tensor::Tensor attention_residual;

    tensor::Tensor ffn_gate;
    tensor::Tensor ffn_up;
    tensor::Tensor ffn_activated;
    tensor::Tensor ffn_down;

    tensor::Tensor rope_cos;
    tensor::Tensor rope_sin;

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

    op::RmsNormOp _norm{TinyLlamaProfile::rms_norm_eps};

    op::LinearOp _output;

    // nullptr 表示尚未成功 prepare。
    std::unique_ptr<TinyLlamaRuntime> _runtime;
};

} // namespace model
