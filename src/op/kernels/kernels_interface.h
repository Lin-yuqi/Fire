#pragma once
#include "Fire/base/base.h"
#include <Fire/tensor/tensor.h>


// -----------------kernel begin------------------
namespace kernel {

typedef void (*AddKernel)(const tensor::Tensor& input1,const tensor::Tensor&input2,
                        tensor::Tensor&output, void* stream);

typedef void (*RMSNormKernel)(const tensor::Tensor& input,const tensor::Tensor&weight,
                        tensor::Tensor&output,const float eps, void* stream);

typedef  void (*RMSNormKernelDim)(const tensor::Tensor& input,const tensor::Tensor&weight,
                        tensor::Tensor&output,const float eps, void* stream);


AddKernel get_add_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType dtype);

RMSNormKernel get_rmsnorm_kernel_dim(base::DeviceType dtype);

}
// -----------------kernel end--------------------