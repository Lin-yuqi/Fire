#include "Fire/model/tinyllama.h"
#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "Fire/model/model_weights.h"
#include "Fire/tensor/tensor.h"
#include <memory>
#include <utility>

namespace model {

base::Status TinyLlamaModel::create(const TinyLlamaWeights& weights, op::OpContext context,
                                    std::unique_ptr<TinyLlamaModel>& output) {
    if (weights.layers.size() != TinyLlamaProfile::num_layers) {
        return base::error::ModelParseError("num_layers doesnt match");
    }
    auto model = std::unique_ptr<TinyLlamaModel>(new TinyLlamaModel(weights));

    if (context._device_type == base::DeviceType::GPU) {
        model->_embedding.to_cuda();
        for (auto& layer : model->_layers) {
            layer.attention_norm.to_cuda();
            layer.wq.to_cuda();
            layer.wk.to_cuda();
            layer.wv.to_cuda();
            layer.ffn_norm.to_cuda();
            layer.w1.to_cuda();
            layer.w2.to_cuda();
            layer.w3.to_cuda();
        }
    }
    model->_norm.to_cuda();
    model->_output.to_cuda();
    output = std::move(model);
    return base::error::Success();
}

TinyLlamaModel::~TinyLlamaModel() {}

const ModelConfig& TinyLlamaModel::config() const noexcept { return TinyLlamaProfile::model; }

base::Status TinyLlamaModel::prepare(int32_t capacity, const op::OpContext& context) {
    // 1. 检查 capacity
    if (capacity > TinyLlamaProfile::model.max_seq_len || capacity <= 0) {
        return base::error::InvalidArgument(
            "capacity must less than max_seq_len and more than zero");
    }
    // 2. 根据 context 拿 allocator
    std::shared_ptr<base::DeviceAllocator> alloc;
    if (context._device_type == base::DeviceType::CPU)
        alloc = base::CPUAllocatorFactory::get_instance();
    else
        alloc = base::GPUAllocatorFactory::get_instance();
    // 3. 创建临时 Runtime
    auto runtime = std::make_unique<TinyLlamaRuntime>();
    // 4. 给所有 Tensor 分配空间

    // hidden states
    runtime->hidden = tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);
    runtime->block_output =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);
    runtime->norm_output =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);

    // QKV
    runtime->query =
        tensor::Tensor(base::DataType::Fp32,
                       {TinyLlamaProfile::num_attention_heads, TinyLlamaProfile::head_dim}, alloc);
    runtime->key = tensor::Tensor(
        base::DataType::Fp32, {TinyLlamaProfile::num_kv_heads, TinyLlamaProfile::head_dim}, alloc);
    runtime->value = tensor::Tensor(
        base::DataType::Fp32, {TinyLlamaProfile::num_kv_heads, TinyLlamaProfile::head_dim}, alloc);

    // attention
    runtime->attention_score = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::num_attention_heads,capacity});
    runtime->attention_output = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::hidden_size},alloc);
    runtime->attention_projected = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::hidden_size},alloc);
    runtime->attention_projected = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::hidden_size},alloc);
    runtime->attention_residual = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::hidden_size},alloc);


    // FFN
    runtime->ffn_gate = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::intermediate_size},alloc);
    runtime->ffn_activated = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::intermediate_size},alloc);
    runtime->ffn_up = tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::intermediate_size},alloc);
    runtime->ffn_down =tensor::Tensor(base::DataType::Fp32,{TinyLlamaProfile::hidden_size},alloc);
    // 5. 分配 KVCache
    runtime->kv_cache.allocate(TinyLlamaProfile::num_layers, capacity, TinyLlamaProfile::num_kv_heads, TinyLlamaProfile::head_dim, alloc);
    // 6. 准备 RoPE cache
    //TODO
    // 7. 全部成功后再发布到 _runtime
    _runtime = std::move(runtime);
    return base::error::Success();
}

base::Status TinyLlamaModel::forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                                     const op::OpContext& context) {}

base::Status TinyLlamaModel::reset(const op::OpContext& ) {
    if (_runtime == nullptr) {
        return base::error::InternalError(
            "TinyLlamaModel has not been prepared");
    }

    _runtime->kv_cache.reset();

    return base::error::Success();
}
TinyLlamaModel::TinyLlamaModel(const TinyLlamaWeights& validated_weights) {

    // embedding 一个参数
    _embedding.reset_param_size(1);
    _embedding.set_param(0, validated_weights.embedding);

    // transformer blocks
    _layers.reserve(validated_weights.layers.size());
    for (const auto& layer : validated_weights.layers) {
        TinyLlamaBlock block;

        block.attention_norm.reset_param_size(1);
        block.attention_norm.set_param(0, layer.attention_norm);

        block.wq.reset_param_size(1);
        block.wq.set_param(0, layer.wq);

        block.wk.reset_param_size(1);
        block.wk.set_param(0, layer.wk);

        block.wv.reset_param_size(1);
        block.wv.set_param(0, layer.wv);

        block.wo.reset_param_size(1);
        block.wo.set_param(0, layer.wo);

        block.ffn_norm.reset_param_size(1);
        block.ffn_norm.set_param(0, layer.ffn_norm);

        block.w1.reset_param_size(1);
        block.w1.set_param(0, layer.w1);

        block.w2.reset_param_size(1);
        block.w2.set_param(0, layer.w2);

        block.w3.reset_param_size(1);
        block.w3.set_param(0, layer.w3);

        _layers.emplace_back(std::move(block));
    }

    // final norm
    _norm.reset_param_size(1);
    _norm.set_param(0, validated_weights.norm);

    // lm head
    _output.reset_param_size(1);
    _output.set_param(0, validated_weights.output);
}
} // namespace model