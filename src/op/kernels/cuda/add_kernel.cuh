#pragma once
#include <Fire/tensor/tensor.h>

// -----------------kernel begin------------------
namespace kernel {
void add_kernel_cu(const tensor::Tensor &input1, const tensor::Tensor &input2,
                   tensor::Tensor &output, void *stream = nullptr);
}
// -----------------kernel end--------------------
