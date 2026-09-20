#include "Fire/model/qwen_loader.h"

#include <utility>

namespace model {

Qwen3Loader::Qwen3Loader(Qwen3Profile profile) : _profile(profile) {}

const Qwen3Profile& Qwen3Loader::profile() const noexcept { return _profile; }

base::Status Qwen3Loader::open(const std::string& path) {
    if (!_profile.is_valid()) {
        return base::error::InvalidArgument("invalid Qwen3 profile");
    }
    return _reader.open(path);
}

base::Status Qwen3Loader::loader_tensor(const std::string& name,
                                        tensor::Tensor& output_tensor) const {
    const TensorInfo* info = _reader.find(name);
    if (info == nullptr) {
        return base::error::ModelParseError("Qwen3 tensor not found: " + name);
    }

    output_tensor = tensor::Tensor(info->dtype, info->dims, _reader.mapped_buffer(),
                                   static_cast<size_t>(info->byte_offset));
    return base::error::Success();
}

base::Status Qwen3Loader::load_tensor(const std::string& name,
                                      const std::vector<int32_t>& expected_dims,
                                      tensor::Tensor& output_tensor) const {
    tensor::Tensor tensor;
    auto status = loader_tensor(name, tensor);
    if (!status) {
        return status;
    }
    if (tensor.data_type() != base::DataType::Fp32 || tensor.dims() != expected_dims) {
        return base::error::ModelParseError("invalid Qwen3 tensor: " + name);
    }
    output_tensor = std::move(tensor);
    return base::error::Success();
}

base::Status Qwen3Loader::load_weights(Qwen3Weights& output_weights) const {
    if (!_profile.is_valid()) {
        return base::error::InvalidArgument("invalid Qwen3 profile");
    }
    if (_reader.tensor_count() != _profile.tensor_count()) {
        return base::error::ModelParseError("Qwen3 tensor count does not match profile");
    }

    Qwen3Weights weights;
    weights.profile = _profile;
    weights.layers.resize(static_cast<size_t>(_profile.num_layers));

    auto status = load_tensor("tok_embeddings.weight",
                              {_profile.model.vocab_size, _profile.hidden_size},
                              weights.embedding);
    if (!status) {
        return status;
    }

    for (int32_t index = 0; index < _profile.num_layers; ++index) {
        auto& layer = weights.layers[static_cast<size_t>(index)];
        const std::string prefix = "layers." + std::to_string(index);

        status = load_tensor(prefix + ".attention_norm.weight", {_profile.hidden_size},
                             layer.attention_norm);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.wq.weight",
                             {_profile.q_dim(), _profile.hidden_size}, layer.wq);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.wk.weight",
                             {_profile.kv_dim(), _profile.hidden_size}, layer.wk);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.wv.weight",
                             {_profile.kv_dim(), _profile.hidden_size}, layer.wv);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.wo.weight",
                             {_profile.hidden_size, _profile.q_dim()}, layer.wo);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.q_norm.weight", {_profile.head_dim},
                             layer.q_norm);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".attention.k_norm.weight", {_profile.head_dim},
                             layer.k_norm);
        if (!status) {
            return status;
        }
        status =
            load_tensor(prefix + ".ffn_norm.weight", {_profile.hidden_size}, layer.ffn_norm);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".feed_forward.w1.weight",
                             {_profile.intermediate_size, _profile.hidden_size}, layer.w1);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".feed_forward.w2.weight",
                             {_profile.hidden_size, _profile.intermediate_size}, layer.w2);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".feed_forward.w3.weight",
                             {_profile.intermediate_size, _profile.hidden_size}, layer.w3);
        if (!status) {
            return status;
        }
    }

    status = load_tensor("norm.weight", {_profile.hidden_size}, weights.norm);
    if (!status) {
        return status;
    }
    status = load_tensor("output.weight", {_profile.model.vocab_size, _profile.hidden_size},
                         weights.output);
    if (!status) {
        return status;
    }

    output_weights = std::move(weights);
    return base::error::Success();
}

} // namespace model
