#pragma once
#include <Fire/tensor/tensor.h>

namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2, float scale,
                      tensor::Tensor& output, void* stream);

void matmul_quant_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            tensor::Tensor& output, int32_t group_size,
                            const tensor::Tensor& scale, const tensor::Tensor& zero_points,
                            void* stream);

} // namespace kernel
