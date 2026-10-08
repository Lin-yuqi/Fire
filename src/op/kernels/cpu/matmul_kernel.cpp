#include "matmul_kernel.h"
#include "bf16.h"
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
    float* output_ptr = output.ptr<float>();

    const int32_t rows = input1.dims_size() == 2 ? input1.get_dim(0) : 1;
    const int32_t input_features = input1.get_dim(input1.dims_size() - 1);
    const int32_t output_features = input2.get_dim(0);
    CHECK_EQ(input2.get_dim(1), input_features);
    CHECK_EQ(output.size(), static_cast<size_t>(rows) * output_features);

    if (input2.data_type() == base::DataType::Bf16) {
        const uint16_t* weight_ptr = input2.ptr<uint16_t>();
        // Decode each weight at use, keeping the stored matrix in BF16.
        for (int32_t row = 0; row < rows; ++row) {
            const float* input_row = input_ptr + static_cast<size_t>(row) * input_features;
            for (int32_t feature = 0; feature < output_features; ++feature) {
                const uint16_t* weight_row =
                    weight_ptr + static_cast<size_t>(feature) * input_features;
                float sum = 0.0f;
                for (int32_t k = 0; k < input_features; ++k) {
                    sum += input_row[k] * bf16_to_fp32(weight_row[k]);
                }
                output_ptr[static_cast<size_t>(row) * output_features + feature] = sum * scale;
            }
        }
        return;
    }

    const float* weight_ptr = input2.ptr<float>();
    // Tensor uses row-major storage while Armadillo uses column-major storage. Map the
    // buffers as transposed views and calculate C^T = W * A^T, where C = A * W^T.
    arma::fmat input_transposed(const_cast<float*>(input_ptr), input_features, rows, false, true);
    arma::fmat weight_transposed(const_cast<float*>(weight_ptr), input_features, output_features,
                                 false, true);
    arma::fmat output_transposed(output_ptr, output_features, rows, false, true);
    output_transposed = scale * weight_transposed.t() * input_transposed;
}

} // namespace kernel
