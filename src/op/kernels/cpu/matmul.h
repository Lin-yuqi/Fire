#pragma once
#include <Fire/tensor/tensor.h>

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,float scale,
                       tensor::Tensor& output, void* stream);
}