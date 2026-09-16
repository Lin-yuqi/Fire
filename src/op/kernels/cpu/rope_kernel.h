#pragma once
#include "Fire/tensor/tensor.h"

namespace kernel {
void sin_cos_cache_kernel_cpu(int head_size, int max_seq_len, float rope_theta,
                              tensor::Tensor& sin_cache, tensor::Tensor& cos_cache, void* stream);

void rope_kernel_cpu(tensor::Tensor& input_q, tensor::Tensor& input_k, const tensor::Tensor& cos,
                     const tensor::Tensor& sin, int32_t pos, void* stream);
} // namespace kernel