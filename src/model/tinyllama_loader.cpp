#include "Fire/base/base.h"
#include "Fire/model/model_weights.h"
#include <Fire/model/tinyllama_loader.h>

namespace model {

base::Status TinyllamaLoader::open(const std::string& path) { return _reader.open(path); }

base::Status TinyllamaLoader::loader_tensor(const std::string& name,
                                            tensor::Tensor& output_tensor) const {
    const TensorInfo* info = _reader.find(name);
    if (info == nullptr) {
        return base::error::ModelParseError("TinyLlama tensor not found: " + name);
    }

    auto buffer = _reader.mapped_buffer();

    output_tensor =
        tensor::Tensor(info->dtype, info->dims, buffer, static_cast<size_t>(info->byte_offset));
    return base::error::Success();
}

base::Status TinyllamaLoader::load_weights(TinyLlamaWeights& output_weights) const {
    if (_reader.tensor_count() != TinyLlamaProfile::tensor_count) {
        return base::error::ModelParseError("tensor count doesnt match");
    }

    TinyLlamaWeights weights;
    weights.layers.resize(TinyLlamaProfile::num_layers);
    // ---------------- embedding ----------------
    auto status = loader_tensor("tok_embeddings.weight", weights.embedding);
    if (!status.ok()) {
        return status;
    }

    if (weights.embedding.data_type() != base::DataType::Fp32 ||
        weights.embedding.dims() != std::vector<int32_t>{TinyLlamaProfile::model.vocab_size,
                                                         TinyLlamaProfile::hidden_size}) {
        return base::error::ModelParseError("invalid tensor: tok_embeddings.weight");
    }
    // ---------------- layers ----------------
    for (int i = 0; i < TinyLlamaProfile::num_layers; i++) {
        auto& layer = weights.layers[i];
        const std::string prefix = "layers." + std::to_string(i);
        status = loader_tensor(prefix + ".attention_norm.weight", layer.attention_norm);
        if (!status.ok()) {
            return status;
        }
        if (layer.attention_norm.data_type() != base::DataType::Fp32 ||
            layer.attention_norm.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".attention_norm.weight");
        }
        status = loader_tensor(prefix + ".attention.wq.weight", layer.wq);
        if (!status.ok()) {
            return status;
        }
        if (layer.wq.data_type() != base::DataType::Fp32 ||
            layer.wq.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size,
                                                    TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".attention.wq.weight");
        }

        status = loader_tensor(prefix + ".attention.wk.weight", layer.wk);
        if (!status.ok()) {
            return status;
        }
        if (layer.wk.data_type() != base::DataType::Fp32 ||
            layer.wk.dims() !=
                std::vector<int32_t>{TinyLlamaProfile::kv_dim, TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".attention.wk.weight");
        }

        status = loader_tensor(prefix + ".attention.wv.weight", layer.wv);
        if (!status.ok()) {
            return status;
        }

        if (layer.wv.data_type() != base::DataType::Fp32 ||
            layer.wv.dims() !=
                std::vector<int32_t>{TinyLlamaProfile::kv_dim, TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".attention.wv.weight");
        }

        status = loader_tensor(prefix + ".attention.wo.weight", layer.wo);
        if (!status.ok()) {
            return status;
        }

        if (layer.wo.data_type() != base::DataType::Fp32 ||
            layer.wo.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size,
                                                    TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".attention.wo.weight");
        }

        status = loader_tensor(prefix + ".ffn_norm.weight", layer.ffn_norm);
        if (!status.ok()) {
            return status;
        }

        if (layer.ffn_norm.data_type() != base::DataType::Fp32 ||
            layer.ffn_norm.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix + ".ffn_norm.weight");
        }

        status = loader_tensor(prefix + ".feed_forward.w1.weight", layer.w1);
        if (!status.ok()) {
            return status;
        }

        if (layer.w1.data_type() != base::DataType::Fp32 ||
            layer.w1.dims() != std::vector<int32_t>{TinyLlamaProfile::intermediate_size,
                                                    TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".feed_forward.w1.weight");
        }

        status = loader_tensor(prefix + ".feed_forward.w2.weight", layer.w2);
        if (!status.ok()) {
            return status;
        }

        if (layer.w2.data_type() != base::DataType::Fp32 ||
            layer.w2.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size,
                                                    TinyLlamaProfile::intermediate_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".feed_forward.w2.weight");
        }

        status = loader_tensor(prefix + ".feed_forward.w3.weight", layer.w3);
        if (!status.ok()) {
            return status;
        }

        if (layer.w3.data_type() != base::DataType::Fp32 ||
            layer.w3.dims() != std::vector<int32_t>{TinyLlamaProfile::intermediate_size,
                                                    TinyLlamaProfile::hidden_size}) {
            return base::error::ModelParseError("invalid tensor: " + prefix +
                                                ".feed_forward.w3.weight");
        }
    }
    // ---------------- final norm ----------------
    status = loader_tensor("norm.weight", weights.norm);
    if (!status.ok()) {
        return status;
    }
    if (weights.norm.data_type() != base::DataType::Fp32 ||
        weights.norm.dims() != std::vector<int32_t>{TinyLlamaProfile::hidden_size}) {
        return base::error::ModelParseError("invalid tensor: norm.weight");
    }
    // ---------------- lm head ----------------
    status = loader_tensor("output.weight", weights.output);
    if (!status.ok()) {
        return status;
    }

    if (weights.output.data_type() != base::DataType::Fp32 ||
        weights.output.dims() != std::vector<int32_t>{TinyLlamaProfile::model.vocab_size,
                                                      TinyLlamaProfile::hidden_size}) {
        return base::error::ModelParseError("invalid tensor: output.weight");
    }

    // 前面 201 个全部成功，才真正修改调用方的 output
    output_weights = std::move(weights);

    return base::error::Success();
}

} // namespace model
