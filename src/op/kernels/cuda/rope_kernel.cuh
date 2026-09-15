#pragma once

#include <Fire/tensor/tensor.h>

namespace kernel {
void rope_kernel_cu(tensor::Tensor& input_q, tensor::Tensor&& input_k, const tensor::Tensor& cos,
                    const tensor::Tensor& sin, int32_t pos);
}