#pragma once
#include "Fire/base/base.h"
#include <Fire/tensor/tensor.h>


// -----------------kernel begin------------------
namespace kernel {

typedef void (*AddKernel)(const tensor::Tensor& input1,const tensor::Tensor&input2,
                        tensor::Tensor&output, void* stream);

                    




AddKernel get_add_kernel(base::DeviceType dtype);

}
// -----------------kernel end--------------------