#include "swiglu_kernel.h"
#include <armadillo>

namespace kernel {
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       tensor::Tensor& output, void*) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK(input1.device_type() == base::DeviceType::CPU);
    CHECK(input2.device_type() == base::DeviceType::CPU);
    CHECK(output.device_type() == base::DeviceType::CPU);

    arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec out_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

    out_vec = input1_vec % (1.0f / (1.0f + arma::exp(-input1_vec))) % input2_vec;
}

} // namespace kernel
