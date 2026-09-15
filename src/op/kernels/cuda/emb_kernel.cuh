#pragma once
#include <Fire/tensor/tensor.h>

// -----------------kernel begin------------------
namespace kernel {
void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   tensor::Tensor& output, void* stream);

} // namespace kernel
