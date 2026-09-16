#pragma once
#include "Fire/tensor/tensor.h"

namespace kernel {

void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       tensor::Tensor& output, void* stream);

}