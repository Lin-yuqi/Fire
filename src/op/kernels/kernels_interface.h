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

typedef void (*MatmulKernelQuant)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                  tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, const tensor::Tensor& zero_points,
                                  void* stream);

typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                tensor::Tensor& output, void* stream);

typedef void (*RoPEKernel)(tensor::Tensor& input_q, tensor::Tensor& input_k,
                           const tensor::Tensor& cos, const tensor::Tensor& sin, int32_t pos,
                           void* stream);

// Each cache has shape [max_seq_len, head_size / 2].
typedef void (*RoPECacheKernel)(int32_t head_size, int32_t max_seq_len, float rope_theta,
                                tensor::Tensor& sin_cache, tensor::Tensor& cos_cache, void* stream);

typedef void (*SwiGLUKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             tensor::Tensor& output, void* stream);

typedef void (*SoftmaxKernel)(const tensor::Tensor& inuput, tensor::Tensor& output, void* stream);

typedef void (*MHAKernel)(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                          const tensor::Tensor& val_cache, tensor::Tensor& score,
                          tensor::Tensor& output, int32_t layer_idx, int32_t pos, void* stream);

AddKernel get_add_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel_dim(base::DeviceType dtype);

MatmulKernel get_matmul_kernel(base::DeviceType dtype);

MatmulKernelQuant get_matmul_quant_kernel(base::DeviceType dtype);

EmbeddingKernel get_embedding_kernel(base::DeviceType dtype);

RoPEKernel get_rope_kernel(base::DeviceType dtype);

RoPECacheKernel get_rope_cache_kernel(base::DeviceType dtype);

SwiGLUKernel get_swiglu_kernel(base::DeviceType dtype);

SoftmaxKernel get_softmax_kernel(base::DeviceType dtype);

MHAKernel get_mha_kernel(base::DeviceType dtype);

} // namespace kernel
// -----------------kernel end--------------------
