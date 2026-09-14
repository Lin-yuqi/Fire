#pragma once

#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "Fire/tensor/tensor.h"

#include <cstdint>
#include <string_view>

namespace model {

// Stable information needed by a text-generation caller. Model-specific
// dimensions belong to the concrete Model Profile, not this common contract.
struct ModelConfig {
    std::string_view profile_name;
    int32_t vocab_size = 0;
    int32_t max_seq_len = 0;
};

// One instance serves one autoregressive sequence. Execution context is
// supplied per call; concrete models own their typed runtime and sequence state.
class Model {
  public:
    virtual ~Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    virtual const ModelConfig& config() const noexcept = 0;

    // Allocate runtime storage for 1 <= capacity <= config().max_seq_len.
    // Success starts an empty sequence; failure preserves the previous runtime.
    // Weights must already match context. This does not load or move weights.
    // The caller must finish outstanding work before replacing runtime storage.
    virtual base::Status prepare(int32_t capacity, const op::OpContext& context) = 0;

    // Consume one token at the zero-based position equal to the valid prefix
    // length. Caller-owned logits must be FP32 [vocab_size] on context's device.
    // Success advances the prefix once; failure leaves the prefix unchanged
    // and does not make logits valid.
    // CUDA success means submission, not completion. Calls must be ordered;
    // the caller owns synchronization and keeps storage alive until completion.
    virtual base::Status forward(int32_t token_id, int32_t pos, tensor::Tensor& logits,
                                 const op::OpContext& context) = 0;

    // Start an empty sequence after prepare, retaining weights and capacity.
    // Does not retain context or release storage used by outstanding work.
    virtual base::Status reset(const op::OpContext& context) = 0;

  protected:
    Model() = default;
};

} // namespace model
