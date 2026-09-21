#pragma once

#include "Fire/model/model.h"
#include "Fire/op/operator.h"
#include "Fire/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace model {

// This header is the shared model-description seam between loaders and model
// assembly. Profiles contain architecture and canonical tensor-set data;
// runtime capacity and payload encodings such as FP32 or INT4 are separate.

// Fixed TinyLlama profile. It is not metadata read from .fire v1 and does not
// describe the wider Llama family.
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

// FP32 tensors retain HF row-major layouts. In particular, Q/K weights are not
// permuted, and RoPE uses the matching half-split layout.
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

// One value type describes every supported dense Qwen3 size. Keeping model
// sizes as values lets one Loader and one Qwen3Model implementation serve both
// 0.6B and 8B without template or subclass duplication.
struct Qwen3Profile {
    ModelConfig model{};
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t head_dim = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    bool tie_word_embeddings = false;

    constexpr int32_t q_dim() const { return num_attention_heads * head_dim; }
    constexpr int32_t kv_dim() const { return num_kv_heads * head_dim; }
    constexpr int32_t kv_groups() const {
        return num_kv_heads == 0 ? 0 : num_attention_heads / num_kv_heads;
    }
    constexpr size_t tensor_count() const {
        // embedding + final norm + output, and eleven tensors per decoder layer.
        return static_cast<size_t>(num_layers) * 11 + 3;
    }

    constexpr bool is_valid() const {
        return !model.profile_name.empty() && model.vocab_size > 0 && model.max_seq_len > 0 &&
               hidden_size > 0 && intermediate_size > 0 && num_layers > 0 &&
               num_attention_heads > 0 && num_kv_heads > 0 && head_dim > 0 &&
               num_attention_heads % num_kv_heads == 0 && head_dim % 2 == 0 &&
               rms_norm_eps > 0.0f && rope_theta > 0.0f;
    }
};

namespace qwen3_profiles {

inline constexpr Qwen3Profile Qwen3_0_6B{
    {"Qwen/Qwen3-0.6B", 151936, 40960},
    1024,
    3072,
    28,
    16,
    8,
    128,
    1e-6f,
    1'000'000.0f,
    true,
};

inline constexpr Qwen3Profile Qwen3_8B{
    {"Qwen/Qwen3-8B", 151936, 40960},
    4096,
    12288,
    36,
    32,
    8,
    128,
    1e-6f,
    1'000'000.0f,
    false,
};

static_assert(Qwen3_0_6B.is_valid());
static_assert(Qwen3_8B.is_valid());
static_assert(Qwen3_0_6B.q_dim() == 2048);
static_assert(Qwen3_8B.q_dim() == Qwen3_8B.hidden_size);

} // namespace qwen3_profiles

// Linear Parameter fields carry either FP32 weights or future packed weights;
// the dimensions below are logical [out_features, in_features] shapes. The
// current .fire v1 loader fills FP32 data in exported row-major layout.
// q_dim is explicit because 0.6B has q_dim=2048 and hidden_size=1024.
struct Qwen3LayerWeights {
    tensor::Tensor attention_norm; // [hidden_size]

    op::Parameter wq; // [q_dim, hidden_size]
    op::Parameter wk; // [kv_dim, hidden_size]
    op::Parameter wv; // [kv_dim, hidden_size]
    op::Parameter wo; // [hidden_size, q_dim]

    tensor::Tensor q_norm;         // [head_dim]
    tensor::Tensor k_norm;         // [head_dim]
    tensor::Tensor ffn_norm;       // [hidden_size]

    op::Parameter w1; // gate: [intermediate_size, hidden_size]
    op::Parameter w2; // down: [hidden_size, intermediate_size]
    op::Parameter w3; // up: [intermediate_size, hidden_size]
};

// Loader result and Qwen3Model assembly input. Carrying the selected profile
// with its tensors prevents accidental 0.6B-weights/8B-profile pairing.
struct Qwen3Weights {
    Qwen3Profile profile{};
    tensor::Tensor embedding; // [vocab_size, hidden_size]
    std::vector<Qwen3LayerWeights> layers;
    tensor::Tensor norm;  // [hidden_size]
    op::Parameter output; // lm_head, logical [vocab_size, hidden_size]
};

} // namespace model
