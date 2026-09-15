#include "matmul_kernel.h"
#include <armadillo>
namespace kernel {

void matmul_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2, float scale,
                       tensor::Tensor& output, void*) {
    CHECK(!input1.is_empty());
    CHECK(!input2.is_empty());
    CHECK(!output.is_empty());
    CHECK(input1.device_type() == base::DeviceType::CPU);
    CHECK(input2.device_type() == base::DeviceType::CPU);
    CHECK(output.device_type() == base::DeviceType::CPU);
    CHECK(input1.dims_size() == 1 || input1.dims_size() == 2);
    CHECK_EQ(input2.dims_size(), 2);

    const float* input_ptr = input1.ptr<float>();
    const float* weight_ptr = input2.ptr<float>();
    float* output_ptr = output.ptr<float>();

    const int32_t rows = input1.dims_size() == 2 ? input1.get_dim(0) : 1;
    const int32_t input_features = input1.get_dim(input1.dims_size() - 1);
    const int32_t output_features = input2.get_dim(0);
    CHECK_EQ(input2.get_dim(1), input_features);
    CHECK_EQ(output.size(), static_cast<size_t>(rows) * output_features);

    // Tensor uses row-major storage while Armadillo uses column-major storage. Map the
    // buffers as transposed views and calculate C^T = W * A^T, where C = A * W^T.
    arma::fmat input_transposed(const_cast<float*>(input_ptr), input_features, rows, false, true);
    arma::fmat weight_transposed(const_cast<float*>(weight_ptr), input_features, output_features,
                                 false, true);
    arma::fmat output_transposed(output_ptr, output_features, rows, false, true);
    output_transposed = scale * weight_transposed.t() * input_transposed;
}

} // namespace kernel
