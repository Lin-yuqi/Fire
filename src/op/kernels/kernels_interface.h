#pragma once
#include "Fire/base/base.h"
#include <Fire/tensor/tensor.h>
#include <cstdint>

// -----------------kernel begin------------------
namespace kernel {

typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          tensor::Tensor& output, void* stream);

typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              tensor::Tensor& output, const float eps, void* stream);

typedef void (*RMSNormKernelDim)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 tensor::Tensor& output, const float eps, void* stream);

typedef void (*MatmulKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             float scale, tensor::Tensor& output, void* stream);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                tensor::Tensor& output, void* stream);

typedef void (*RoPEKernel)(tensor::Tensor& input_q, tensor::Tensor&& input_k,
                           const tensor::Tensor& cos, const tensor::Tensor& sin, int32_t pos);

AddKernel get_add_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel_dim(base::DeviceType dtype);

MatmulKernel get_matmul_kernel(base::DeviceType dtype);

EmbeddingKernel get_embedding_kernel(base::DeviceType dtype);

RoPEKernel get_rope_kernel(base::DeviceType dtype);

} // namespace kernel
// -----------------kernel end--------------------