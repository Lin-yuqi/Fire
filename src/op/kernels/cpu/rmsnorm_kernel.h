#pragma once
#include <Fire/tensor/tensor.h>

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        tensor::Tensor& output,const float eps, void* stream);
}