#pragma once
#include <Fire/tensor/tensor.h>

namespace kernel {
void mha_kernel_cu(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                          const tensor::Tensor& val_cache, tensor::Tensor& score,
                          tensor::Tensor& output,int32_t layer_idx,  int32_t pos, void* stream);
} // namespace kernel