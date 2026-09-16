#pragma once
#include "Fire/tensor/tensor.h"

namespace kernel {
void softmax_kernel_cpu(const tensor::Tensor& inuput, tensor::Tensor& output, void* stream);
}