#pragma once

#include "Fire/model/model_weights.h"
#include "Fire/op/linear.h"
#include "Fire/op/rmsnorm.h"

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

// TODO: implement TinyLlamaModel and its typed Runtime after structured weight
// loading, KVCache and the missing operators. Do not add device/stream state
// to this block or expose a generic KVCache requirement through Model.

} // namespace model
