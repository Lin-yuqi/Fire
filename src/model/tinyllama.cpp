#include "Fire/model/tinyllama.h"
#include "Fire/base/base.h"
#include "Fire/model/model_weights.h"
#include "Fire/tensor/tensor.h"
#include "../op/kernels/kernels_interface.h"
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
            layer.wo.to_cuda();
            layer.ffn_norm.to_cuda();
            layer.w1.to_cuda();
            layer.w2.to_cuda();
            layer.w3.to_cuda();
        }
        model->_norm.to_cuda();
        model->_output.to_cuda();
    }
    output = std::move(model);
    return base::error::Success();
}

TinyLlamaModel::~TinyLlamaModel() = default;

const ModelConfig& TinyLlamaModel::config() const noexcept { return TinyLlamaProfile::model; }

base::Status TinyLlamaModel::prepare(int32_t capacity, const op::OpContext& context) {
    // 1. 检查 capacity
    if (capacity > TinyLlamaProfile::model.max_seq_len || capacity <= 0) {
        return base::error::InvalidArgument(
            "capacity must less than max_seq_len and more than zero");
    }
    if (context._device_type != base::DeviceType::CPU &&
        context._device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("prepare requires a CPU or GPU device");
    }
    if (context._allocator == nullptr) {
        return base::error::InvalidArgument("prepare requires an allocator");
    }
    if (context._allocator->device_type() != context._device_type) {
        return base::error::InvalidArgument(
            "prepare allocator device does not match the context device");
    }
    const auto parameters_match_device = [&context](const op::ParamOperator& param_op) {
        for (size_t i = 0; i < param_op.param_size(); ++i) {
            if (param_op.get_param(i)._data.device_type() != context._device_type) {
                return false;
            }
        }
        return true;
    };
    if (!parameters_match_device(_embedding) || !parameters_match_device(_norm) ||
        !parameters_match_device(_output)) {
        return base::error::InvalidArgument(
            "prepare context does not match the model weight device");
    }
    for (const auto& layer : _layers) {
        const op::ParamOperator* layer_ops[] = {
            &layer.attention_norm, &layer.wq, &layer.wk, &layer.wv, &layer.wo,
            &layer.ffn_norm,       &layer.w1, &layer.w2, &layer.w3,
        };
        for (const auto* layer_op : layer_ops) {
            if (!parameters_match_device(*layer_op)) {
                return base::error::InvalidArgument(
                    "prepare context does not match the model weight device");
            }
        }
    }
    // 2. 根据 context 拿 allocator
    auto alloc = context._allocator;
    // 3. 创建临时 Runtime
    auto runtime = std::make_unique<TinyLlamaRuntime>();
    // 4. 给所有 Tensor 分配空间

    // token input
    runtime->token = tensor::Tensor(base::DataType::int32, {1}, alloc);

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
    runtime->attention_score = tensor::Tensor(
        base::DataType::Fp32, {TinyLlamaProfile::num_attention_heads, capacity}, alloc);
    runtime->attention_output =
        tensor::Tensor(base::DataType::Fp32,
                       {TinyLlamaProfile::num_attention_heads, TinyLlamaProfile::head_dim}, alloc);
    runtime->attention_projected =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);
    runtime->attention_residual =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);

    // FFN
    runtime->ffn_gate =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::intermediate_size}, alloc);
    runtime->ffn_activated =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::intermediate_size}, alloc);
    runtime->ffn_up =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::intermediate_size}, alloc);
    runtime->ffn_down =
        tensor::Tensor(base::DataType::Fp32, {TinyLlamaProfile::hidden_size}, alloc);
    // 5. 分配 KVCache
    auto status = runtime->kv_cache.allocate(TinyLlamaProfile::num_layers, capacity,
                                             TinyLlamaProfile::num_kv_heads,
                                             TinyLlamaProfile::head_dim, alloc);
    if (!status) {
        return status;
    }
    // 6. 准备 RoPE cache
    // {max_seq_len, head_size / 2}，每对前后半区元素共用一份 sin/cos。
    runtime->rope_cos = tensor::Tensor(
        base::DataType::Fp32, {TinyLlamaProfile::model.max_seq_len, TinyLlamaProfile::head_dim / 2},
        alloc);
    runtime->rope_sin = tensor::Tensor(
        base::DataType::Fp32, {TinyLlamaProfile::model.max_seq_len, TinyLlamaProfile::head_dim / 2},
        alloc);
    kernel::get_rope_cache_kernel(context._device_type)(
        TinyLlamaProfile::head_dim, TinyLlamaProfile::model.max_seq_len,
        TinyLlamaProfile::rope_theta, runtime->rope_sin, runtime->rope_cos, context._stream);

    // 7. 全部成功后再发布到 _runtime
    _runtime = std::move(runtime);
    return base::error::Success();
}

base::Status TinyLlamaModel::forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                                     const op::OpContext& context) {
    if (_runtime == nullptr) {
        return base::error::InternalError("TinyLlamaModel has not been prepared");
    }
    if (token_id < 0 || token_id >= TinyLlamaProfile::model.vocab_size) {
        return base::error::InvalidArgument("token_id is out of vocabulary range");
    }
    if (pos != _runtime->kv_cache.length() || pos < 0 || pos >= _runtime->kv_cache.capacity()) {
        return base::error::InvalidArgument("pos does not match the current sequence length");
    }

    if (context._device_type != base::DeviceType::CPU &&
        context._device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("forward requires a CPU or GPU device");
    }
    if (_runtime->token.device_type() != context._device_type) {
        return base::error::InvalidArgument(
            "forward context does not match the prepared runtime device");
    }

    //----------------------------------------------
    //                  embedding
    //----------------------------------------------
    if (context._device_type == base::DeviceType::CPU) {
        _runtime->token.ptr<int32_t>()[0] = token_id;
    } else {
        const auto cuda_status =
            cudaMemcpyAsync(_runtime->token.ptr<int32_t>(), &token_id, sizeof(token_id),
                            cudaMemcpyHostToDevice, context._stream);
        if (cuda_status != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(cuda_status));
        }
    }

    auto status = _embedding.forward(_runtime->token, _runtime->hidden, context);
    if (!status) {
        return status;
    }

    //----------------------------------------------
    //                  layer
    //----------------------------------------------
    for (int i = 0; i < TinyLlamaProfile::num_layers; i++) {
        auto& layer = _layers[i];
        // attention norm
        status = layer.attention_norm.forward(_runtime->hidden, _runtime->norm_output, context);
        if (!status)
            return status;

        // wq  wk  wv
        _runtime->query.reshape({TinyLlamaProfile::hidden_size});
        status = layer.wq.forward(_runtime->norm_output, _runtime->query, context);
        _runtime->query.reshape(
            {TinyLlamaProfile::num_attention_heads, TinyLlamaProfile::head_dim});
        if (!status)
            return status;
        _runtime->key.reshape({TinyLlamaProfile::num_kv_heads * TinyLlamaProfile::head_dim});
        status = layer.wk.forward(_runtime->norm_output, _runtime->key, context);
        _runtime->key.reshape({TinyLlamaProfile::num_kv_heads, TinyLlamaProfile::head_dim});
        if (!status)
            return status;
        _runtime->value.reshape({TinyLlamaProfile::num_kv_heads * TinyLlamaProfile::head_dim});
        status = layer.wv.forward(_runtime->norm_output, _runtime->value, context);
        _runtime->value.reshape({TinyLlamaProfile::num_kv_heads, TinyLlamaProfile::head_dim});
        if (!status)
            return status;

        // rope
        status = _rope.forward(_runtime->query, _runtime->key, _runtime->rope_cos,
                               _runtime->rope_sin, pos, context);
        if (!status)
            return status;

        // insert kv cache
        status = _runtime->kv_cache.write(i, pos, _runtime->key, _runtime->value, context._stream);
        if (!status)
            return status;
        // mha
        _runtime->attention_output.reshape(
            {TinyLlamaProfile::num_attention_heads, TinyLlamaProfile::head_dim});
        status =
            _mha.forward(_runtime->query, _runtime->kv_cache.key(), _runtime->kv_cache.value(),
                         _runtime->attention_score, _runtime->attention_output, i, pos, context);
        if (!status)
            return status;
        // wo
        _runtime->attention_output.reshape({TinyLlamaProfile::hidden_size});
        status =
            layer.wo.forward(_runtime->attention_output, _runtime->attention_projected, context);
        if (!status)
            return status;

        // 残差连接
        status = _add.forward(_runtime->hidden, _runtime->attention_projected,
                              _runtime->attention_residual, context);
        if (!status)
            return status;

        // ffn
        status =
            layer.ffn_norm.forward(_runtime->attention_residual, _runtime->norm_output, context);
        if (!status)
            return status;

        // swiglu
        // x1
        status = layer.w1.forward(_runtime->norm_output, _runtime->ffn_gate, context);
        if (!status)
            return status;
        // x2
        status = layer.w3.forward(_runtime->norm_output, _runtime->ffn_up, context);
        if (!status)
            return status;

        status =
            _swiglu.forward(_runtime->ffn_gate, _runtime->ffn_up, _runtime->ffn_activated, context);
        if (!status)
            return status;

        // x3
        status = layer.w2.forward(_runtime->ffn_activated, _runtime->ffn_down, context);
        if (!status)
            return status;

        // 再次残差连接
        status = _add.forward(_runtime->attention_residual, _runtime->ffn_down,
                              _runtime->block_output, context);
        if (!status)
            return status;
        std::swap(_runtime->hidden, _runtime->block_output);
    }
    //----------------------------------------------
    //                  final_rmsnorm
    //----------------------------------------------
    status = _norm.forward(_runtime->hidden, _runtime->norm_output, context);
    if (!status)
        return status;
    //----------------------------------------------
    //                  LM HEAD
    //----------------------------------------------
    status = _output.forward(_runtime->norm_output, logits, context);
    if (!status)
        return status;
    return _runtime->kv_cache.commit(pos);
}

base::Status TinyLlamaModel::reset(const op::OpContext&) {
    if (_runtime == nullptr) {
        return base::error::InternalError("TinyLlamaModel has not been prepared");
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
