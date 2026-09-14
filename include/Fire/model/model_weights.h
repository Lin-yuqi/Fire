#pragma once

#include "Fire/model/model.h"
#include "Fire/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace model {

// The fixed C++ profile shared by the Loader and model assembly. This is not
// metadata read from .fire v1 and does not describe the entire Llama family.
struct TinyLlamaProfile {
    inline static constexpr ModelConfig model{
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 32000, 2048};

    static constexpr int32_t hidden_size = 2048;
    static constexpr int32_t intermediate_size = 5632;
    static constexpr int32_t num_layers = 22;
    static constexpr int32_t num_attention_heads = 32;
    static constexpr int32_t num_kv_heads = 4;
    static constexpr int32_t head_dim = hidden_size / num_attention_heads;
    static constexpr int32_t kv_dim = num_kv_heads * head_dim;
    static constexpr int32_t kv_groups = num_attention_heads / num_kv_heads;

    static constexpr float rms_norm_eps = 1e-5f;
    static constexpr float rope_theta = 10000.0f;
    static constexpr size_t tensor_count = num_layers * 9 + 3;
};

// FP32 tensors retain HF row-major layouts. In particular, Q/K weights are
// not permuted: the future RoPE operator must use the matching half-split layout.
struct TinyLlamaLayerWeights {
    tensor::Tensor attention_norm; // [hidden_size]
    tensor::Tensor wq;             // [hidden_size, hidden_size]
    tensor::Tensor wk;             // [kv_dim, hidden_size]
    tensor::Tensor wv;             // [kv_dim, hidden_size]
    tensor::Tensor wo;             // [hidden_size, hidden_size]
    tensor::Tensor ffn_norm;       // [hidden_size]
    tensor::Tensor w1;             // gate: [intermediate_size, hidden_size]
    tensor::Tensor w2;             // down: [hidden_size, intermediate_size]
    tensor::Tensor w3;             // up:   [intermediate_size, hidden_size]
};

// Assembly input, not a second mutable parameter store inside the model.
// Loader views share mmap ownership; move weights to the execution device
// before binding them into ParamOperator's existing Parameter representation.
struct TinyLlamaWeights {
    tensor::Tensor embedding; // [vocab_size, hidden_size]
    std::vector<TinyLlamaLayerWeights> layers;
    tensor::Tensor norm;      // [hidden_size]
    tensor::Tensor output;    // [vocab_size, hidden_size], not tied to embedding
};

} // namespace model
