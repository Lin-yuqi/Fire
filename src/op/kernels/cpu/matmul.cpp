#include "matmul.h"
#include <armadillo>
namespace kernel {

void matmul_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2, float scale,
                       tensor::Tensor& output, void* stream) {
    CHECK(input1.is_empty() == false);
    CHECK(input2.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input1.device_type() == base::DeviceType::CPU);
    CHECK(input2.device_type() == base::DeviceType::CPU);
    CHECK(output.device_type() == base::DeviceType::CPU);

    const float* input_ptr = input1.ptr<float>();
    const float* weight_ptr = input2.ptr<float>();
    const float* output_ptr = output.ptr<float>();

    int32_t in_dim1 = 1;
    int32_t in_dim0 = 1;
    if (input1.dims_size() == 2) {
        in_dim0 = input1.get_dim(0);
        in_dim1 = input1.get_dim(1);
    } else if (input1.dims_size() == 1) {
        in_dim0 = input1.get_dim(0);
    } else {
        LOG(FATAL) << "The input1 tensor has a wrong dim size.";
    }

    CHECK_EQ(input2.dims_size(), 2);
    const int32_t wei_dim0 = input2.get_dim(0);
    const int32_t wei_dim1 = input2.get_dim(1);
    CHECK_EQ(in_dim0, wei_dim1);

    CHECK_EQ(output.size(), wei_dim0 * in_dim1);
    arma::fmat input_mat(const_cast<float*>(input_ptr), in_dim1, in_dim0, false, true);
    arma::fmat weight_mat(const_cast<float*>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::fmat output_mat(const_cast<float*>(output_ptr), in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;
}

} // namespace kernel