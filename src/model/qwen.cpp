#include "Fire/model/qwen.h"

#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "../op/kernels/kernels_interface.h"

#include <cuda_runtime.h>

#include <memory>
#include <utility>

namespace model {
namespace {

bool parameters_match_device(const op::ParamOperator& operation,
                             base::DeviceType device_type) {
    for (size_t index = 0; index < operation.param_size(); ++index) {
        const auto& parameter = operation.get_param(index);
        if (parameter._data.device_type() != device_type ||
            (!parameter._scales.is_empty() && parameter._scales.device_type() != device_type) ||
            (!parameter._zero_points.is_empty() &&
             parameter._zero_points.device_type() != device_type)) {
            return false;
        }
    }
    return true;
}

} // namespace

base::Status Qwen3Model::create(const Qwen3Weights& weights, const op::OpContext& context,
                                std::unique_ptr<Qwen3Model>& output) {
    if (!weights.profile.is_valid()) {
        return base::error::InvalidArgument("invalid Qwen3 profile");
    }
    if (weights.layers.size() != static_cast<size_t>(weights.profile.num_layers)) {
        return base::error::ModelParseError("Qwen3 layer count does not match profile");
    }
    if (context._device_type != base::DeviceType::CPU &&
        context._device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("Qwen3 create requires a CPU or GPU device");
    }

    auto model = std::unique_ptr<Qwen3Model>(new Qwen3Model(weights));
    if (context._device_type == base::DeviceType::GPU) {
        model->_embedding.to_cuda();
        for (auto& layer : model->_layers) {
            layer.attention_norm.to_cuda();
            layer.wq.to_cuda();
            layer.wk.to_cuda();
            layer.wv.to_cuda();
            layer.wo.to_cuda();
            layer.q_norm.to_cuda();
            layer.k_norm.to_cuda();
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

Qwen3Model::~Qwen3Model() = default;

const ModelConfig& Qwen3Model::config() const noexcept { return _profile.model; }

base::Status Qwen3Model::prepare(int32_t capacity, const op::OpContext& context) {
    if (capacity <= 0 || capacity > _profile.model.max_seq_len) {
        return base::error::InvalidArgument(
            "Qwen3 capacity must be positive and not exceed max_seq_len");
    }
    if (context._device_type != base::DeviceType::CPU &&
        context._device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("Qwen3 prepare requires a CPU or GPU device");
    }
    if (context._allocator == nullptr ||
        context._allocator->device_type() != context._device_type) {
        return base::error::InvalidArgument(
            "Qwen3 prepare requires an allocator matching the context device");
    }
    if (!parameters_match_device(_embedding, context._device_type) ||
        !parameters_match_device(_norm, context._device_type) ||
        !parameters_match_device(_output, context._device_type)) {
        return base::error::InvalidArgument(
            "Qwen3 prepare context does not match the model weight device");
    }
    for (const auto& layer : _layers) {
        const op::ParamOperator* operations[] = {
            &layer.attention_norm, &layer.wq,      &layer.wk,       &layer.wv,
            &layer.wo,             &layer.q_norm,  &layer.k_norm,   &layer.ffn_norm,
            &layer.w1,             &layer.w2,      &layer.w3,
        };
        for (const auto* operation : operations) {
            if (!parameters_match_device(*operation, context._device_type)) {
                return base::error::InvalidArgument(
                    "Qwen3 prepare context does not match the model weight device");
            }
        }
    }

    auto runtime = std::make_unique<Qwen3Runtime>();
    auto allocator = context._allocator;
    runtime->token = tensor::Tensor(base::DataType::Int32, {1}, allocator);
    runtime->hidden = tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);
    runtime->block_output =
        tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);
    runtime->norm_output =
        tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);

    runtime->query = tensor::Tensor(
        base::DataType::Fp32, {_profile.num_attention_heads, _profile.head_dim}, allocator);
    runtime->key = tensor::Tensor(
        base::DataType::Fp32, {_profile.num_kv_heads, _profile.head_dim}, allocator);
    runtime->value = tensor::Tensor(
        base::DataType::Fp32, {_profile.num_kv_heads, _profile.head_dim}, allocator);
    runtime->attention_score = tensor::Tensor(
        base::DataType::Fp32, {_profile.num_attention_heads, capacity}, allocator);
    runtime->attention_output = tensor::Tensor(
        base::DataType::Fp32, {_profile.num_attention_heads, _profile.head_dim}, allocator);
    runtime->attention_projected =
        tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);
    runtime->attention_residual =
        tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);

    runtime->ffn_gate =
        tensor::Tensor(base::DataType::Fp32, {_profile.intermediate_size}, allocator);
    runtime->ffn_up =
        tensor::Tensor(base::DataType::Fp32, {_profile.intermediate_size}, allocator);
    runtime->ffn_activated =
        tensor::Tensor(base::DataType::Fp32, {_profile.intermediate_size}, allocator);
    runtime->ffn_down =
        tensor::Tensor(base::DataType::Fp32, {_profile.hidden_size}, allocator);

    auto status = runtime->kv_cache.allocate(_profile.num_layers, capacity,
                                             _profile.num_kv_heads, _profile.head_dim, allocator);
    if (!status) {
        return status;
    }
    runtime->rope_cos = tensor::Tensor(
        base::DataType::Fp32, {_profile.model.max_seq_len, _profile.head_dim / 2}, allocator);
    runtime->rope_sin = tensor::Tensor(
        base::DataType::Fp32, {_profile.model.max_seq_len, _profile.head_dim / 2}, allocator);
    kernel::get_rope_cache_kernel(context._device_type)(
        _profile.head_dim, _profile.model.max_seq_len, _profile.rope_theta, runtime->rope_sin,
        runtime->rope_cos, context._stream);

    _runtime = std::move(runtime);
    return base::error::Success();
}

base::Status Qwen3Model::forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                                 const op::OpContext& context) {
    if (_runtime == nullptr) {
        return base::error::InternalError("Qwen3Model has not been prepared");
    }
    if (token_id < 0 || token_id >= _profile.model.vocab_size) {
        return base::error::InvalidArgument("Qwen3 token_id is out of vocabulary range");
    }
    if (pos != _runtime->kv_cache.length() || pos < 0 ||
        pos >= _runtime->kv_cache.capacity()) {
        return base::error::InvalidArgument(
            "Qwen3 position does not match the current sequence length");
    }
    if (context._device_type != base::DeviceType::CPU &&
        context._device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("Qwen3 forward requires a CPU or GPU device");
    }
    if (_runtime->token.device_type() != context._device_type) {
        return base::error::InvalidArgument(
            "Qwen3 forward context does not match the prepared runtime device");
    }

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

    for (int32_t index = 0; index < _profile.num_layers; ++index) {
        auto& layer = _layers[static_cast<size_t>(index)];

        status = layer.attention_norm.forward(_runtime->hidden, _runtime->norm_output, context);
        if (!status) {
            return status;
        }

        _runtime->query.reshape({_profile.q_dim()});
        status = layer.wq.forward(_runtime->norm_output, _runtime->query, context);
        _runtime->query.reshape({_profile.num_attention_heads, _profile.head_dim});
        if (!status) {
            return status;
        }

        _runtime->key.reshape({_profile.kv_dim()});
        status = layer.wk.forward(_runtime->norm_output, _runtime->key, context);
        _runtime->key.reshape({_profile.num_kv_heads, _profile.head_dim});
        if (!status) {
            return status;
        }

        _runtime->value.reshape({_profile.kv_dim()});
        status = layer.wv.forward(_runtime->norm_output, _runtime->value, context);
        _runtime->value.reshape({_profile.num_kv_heads, _profile.head_dim});
        if (!status) {
            return status;
        }

        status = layer.q_norm.forward(_runtime->query, _runtime->query, context);
        if (!status) {
            return status;
        }
        status = layer.k_norm.forward(_runtime->key, _runtime->key, context);
        if (!status) {
            return status;
        }

        status = _rope.forward(_runtime->query, _runtime->key, _runtime->rope_cos,
                               _runtime->rope_sin, pos, context);
        if (!status) {
            return status;
        }

        status = _runtime->kv_cache.write(index, pos, _runtime->key, _runtime->value,
                                          context._stream);
        if (!status) {
            return status;
        }
        status = _mha.forward(_runtime->query, _runtime->kv_cache.key(),
                              _runtime->kv_cache.value(), _runtime->attention_score,
                              _runtime->attention_output, index, pos, context);
        if (!status) {
            return status;
        }

        _runtime->attention_output.reshape({_profile.q_dim()});
        status =
            layer.wo.forward(_runtime->attention_output, _runtime->attention_projected, context);
        _runtime->attention_output.reshape(
            {_profile.num_attention_heads, _profile.head_dim});
        if (!status) {
            return status;
        }

        status = _add.forward(_runtime->hidden, _runtime->attention_projected,
                              _runtime->attention_residual, context);
        if (!status) {
            return status;
        }

        status =
            layer.ffn_norm.forward(_runtime->attention_residual, _runtime->norm_output, context);
        if (!status) {
            return status;
        }
        status = layer.w1.forward(_runtime->norm_output, _runtime->ffn_gate, context);
        if (!status) {
            return status;
        }
        status = layer.w3.forward(_runtime->norm_output, _runtime->ffn_up, context);
        if (!status) {
            return status;
        }
        status =
            _swiglu.forward(_runtime->ffn_gate, _runtime->ffn_up, _runtime->ffn_activated, context);
        if (!status) {
            return status;
        }
        status = layer.w2.forward(_runtime->ffn_activated, _runtime->ffn_down, context);
        if (!status) {
            return status;
        }
        status = _add.forward(_runtime->attention_residual, _runtime->ffn_down,
                              _runtime->block_output, context);
        if (!status) {
            return status;
        }
        std::swap(_runtime->hidden, _runtime->block_output);
    }

    status = _norm.forward(_runtime->hidden, _runtime->norm_output, context);
    if (!status) {
        return status;
    }
    status = _output.forward(_runtime->norm_output, logits, context);
    if (!status) {
        return status;
    }
    return _runtime->kv_cache.commit(pos);
}

base::Status Qwen3Model::reset(const op::OpContext&) {
    if (_runtime == nullptr) {
        return base::error::InternalError("Qwen3Model has not been prepared");
    }
    _runtime->kv_cache.reset();
    return base::error::Success();
}

Qwen3Model::Qwen3Model(const Qwen3Weights& validated_weights)
    : _profile(validated_weights.profile), _norm(_profile.rms_norm_eps) {
    _embedding.reset_param_size(1);
    _embedding.set_param(0, validated_weights.embedding);

    _layers.reserve(validated_weights.layers.size());
    for (const auto& weights : validated_weights.layers) {
        Qwen3Block layer(_profile.rms_norm_eps);
        layer.attention_norm.reset_param_size(1);
        layer.attention_norm.set_param(0, weights.attention_norm);
        layer.wq.reset_param_size(1);
        layer.wq.set_param(0, weights.wq);
        layer.wk.reset_param_size(1);
        layer.wk.set_param(0, weights.wk);
        layer.wv.reset_param_size(1);
        layer.wv.set_param(0, weights.wv);
        layer.wo.reset_param_size(1);
        layer.wo.set_param(0, weights.wo);
        layer.q_norm.reset_param_size(1);
        layer.q_norm.set_param(0, weights.q_norm);
        layer.k_norm.reset_param_size(1);
        layer.k_norm.set_param(0, weights.k_norm);
        layer.ffn_norm.reset_param_size(1);
        layer.ffn_norm.set_param(0, weights.ffn_norm);
        layer.w1.reset_param_size(1);
        layer.w1.set_param(0, weights.w1);
        layer.w2.reset_param_size(1);
        layer.w2.set_param(0, weights.w2);
        layer.w3.reset_param_size(1);
        layer.w3.set_param(0, weights.w3);
        _layers.emplace_back(std::move(layer));
    }

    _norm.reset_param_size(1);
    _norm.set_param(0, validated_weights.norm);
    _output.reset_param_size(1);
    _output.set_param(0, validated_weights.output);
}

} // namespace model
