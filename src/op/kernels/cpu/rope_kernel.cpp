#include "rope_kernel.h"

namespace kernel {
void rope_kernel_cpu(tensor::Tensor& input_q, tensor::Tensor&& input_k, const tensor::Tensor& cos,
                     const tensor::Tensor& sin, int32_t pos) {}
} // namespace kernel