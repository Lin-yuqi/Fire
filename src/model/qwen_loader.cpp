#include "Fire/model/qwen_loader.h"
#include "Fire/base/base.h"

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

base::Status Qwen3Loader::load_tensor(const std::string& name,
                                      const std::vector<int32_t>& expected_dims,
                                      op::Parameter& output_parameter) const {
    // .qweight .scales .zero_points int4
    // .weight FP32
    
    op::Parameter parameter;
    if (_reader.find(name) != nullptr) {
        if (_reader.tensor_count() != _profile.tensor_count()) {
            return base::error::ModelParseError("unexpected FP32 Qwen3 linear tensor: " + name);
        }
        auto status = load_tensor(name, expected_dims, parameter._data);
        if (!status) {
            return status;
        }
    } else {
        if (_reader.tensor_count() == _profile.tensor_count()) {
            return base::error::ModelParseError("Qwen3 tensor not found: " + name);
        }
        constexpr int32_t group_size = 128;
        constexpr char weight_suffix[] = ".weight";
        if (name.size() < sizeof(weight_suffix) - 1 ||
            name.compare(name.size() - (sizeof(weight_suffix) - 1),
                         sizeof(weight_suffix) - 1, weight_suffix) != 0 ||
            expected_dims.size() != 2 || expected_dims[1] % group_size != 0) {
            return base::error::ModelParseError("invalid Qwen3 quantized tensor: " + name);
        }

        const std::string prefix = name.substr(0, name.size() - (sizeof(weight_suffix) - 1));
        const std::string qweight_name = prefix + ".qweight";
        const std::string scales_name = prefix + ".scales";
        const std::string zero_points_name = prefix + ".zero_points";
        const TensorInfo* qweight = _reader.find(qweight_name);
        const TensorInfo* scales = _reader.find(scales_name);
        const TensorInfo* zero_points = _reader.find(zero_points_name);
        if (qweight == nullptr || scales == nullptr || zero_points == nullptr) {
            return base::error::ModelParseError("Qwen3 quantized tensor not found: " + name);
        }

        const std::vector<int32_t> packed_dims{expected_dims[0], expected_dims[1] / 2};
        const std::vector<int32_t> group_dims{expected_dims[0], expected_dims[1] / group_size};
        if (qweight->dtype != base::DataType::UInt8 || qweight->dims != packed_dims ||
            qweight->quantization_kind != QuantizationKind::Int4GroupWise ||
            qweight->group_size != group_size || scales->dtype != base::DataType::Fp32 ||
            scales->dims != group_dims || scales->quantization_kind != QuantizationKind::None ||
            zero_points->dtype != base::DataType::UInt8 || zero_points->dims != group_dims ||
            zero_points->quantization_kind != QuantizationKind::None) {
            return base::error::ModelParseError("invalid Qwen3 quantized tensor: " + name);
        }

        auto status = loader_tensor(qweight_name, parameter._data);
        if (!status) {
            return status;
        }
        status = loader_tensor(scales_name, parameter._scales);
        if (!status) {
            return status;
        }
        status = loader_tensor(zero_points_name, parameter._zero_points);
        if (!status) {
            return status;
        }
        parameter._quant_config._quant_type = op::QuantType::Int4GroupWise;
        parameter._quant_config._group_size = group_size;
        parameter._quant_config._symmetric = false;
    }

    output_parameter = std::move(parameter);
    return base::error::Success();
}

base::Status Qwen3Loader::load_weights(Qwen3Weights& output_weights) const {
    if (!_profile.is_valid()) {
        return base::error::InvalidArgument("invalid Qwen3 profile");
    }
    // INT4 replaces each of the seven linear weights per layer and the output
    // weight with a qweight/scales/zero_points triplet.
    const size_t quantized_tensor_count =
        _profile.tensor_count() + 2 * (static_cast<size_t>(_profile.num_layers) * 7 + 1);
    if (_reader.tensor_count() != _profile.tensor_count() &&
        _reader.tensor_count() != quantized_tensor_count) {
        return base::error::ModelParseError("Qwen3 tensor count does not match profile");
    }

    Qwen3Weights weights;
    weights.profile = _profile;
    weights.layers.resize(static_cast<size_t>(_profile.num_layers));

    auto status = load_tensor("tok_embeddings.weight",
                              {_profile.model.vocab_size, _profile.hidden_size}, weights.embedding);
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
        status =
            load_tensor(prefix + ".attention.q_norm.weight", {_profile.head_dim}, layer.q_norm);
        if (!status) {
            return status;
        }
        status =
            load_tensor(prefix + ".attention.k_norm.weight", {_profile.head_dim}, layer.k_norm);
        if (!status) {
            return status;
        }
        status = load_tensor(prefix + ".ffn_norm.weight", {_profile.hidden_size}, layer.ffn_norm);
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
