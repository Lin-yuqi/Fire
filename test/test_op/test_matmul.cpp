#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/linear.h"
#include "Fire/op/matmul.h"
#include "Fire/tensor/tensor.h"

namespace {

tensor::Tensor make_cpu_tensor(std::vector<int32_t> dims,
                               const std::vector<float>& values) {
    tensor::Tensor tensor(base::DataType::Fp32, std::move(dims),
                          base::CPUAllocatorFactory::get_instance());
    std::copy(values.begin(), values.end(), tensor.ptr<float>());
    return tensor;
}

op::OpContext cpu_context() {
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    return context;
}

TEST(matmul_test, cpu_vector_with_scale) {
    const auto input = make_cpu_tensor({3}, {1.f, 2.f, 3.f});
    const auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    auto output = make_cpu_tensor({2}, {0.f, 0.f});

    op::MatmulOp matmul;
    const auto status = matmul.forward(input, weight, 0.5f, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 7.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 2.5f);
}

TEST(matmul_test, cpu_matrix_with_scale) {
    const auto input = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    const auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    auto output = make_cpu_tensor({2, 2}, {0.f, 0.f, 0.f, 0.f});

    op::MatmulOp matmul;
    const auto status = matmul.forward(input, weight, 0.5f, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 7.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 2.5f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[2], 16.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[3], 4.f);
}

TEST(linear_test, cpu_vector_with_default_scale_and_no_bias) {
    const auto input = make_cpu_tensor({3}, {1.f, 2.f, 3.f});
    const auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    auto output = make_cpu_tensor({2}, {0.f, 0.f});

    op::LinearOp linear;
    linear.reset_param_size(1);
    linear.get_param(0)._data = weight;

    const auto status = linear.forward(input, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 14.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 5.f);
}

TEST(linear_test, cpu_vector_with_custom_scale_and_bias) {
    const auto input = make_cpu_tensor({3}, {1.f, 2.f, 3.f});
    const auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    const auto bias = make_cpu_tensor({2}, {1.f, -2.f});
    auto output = make_cpu_tensor({2}, {0.f, 0.f});

    op::LinearOp linear(0.5f);
    linear.reset_param_size(2);
    linear.get_param(0)._data = weight;
    linear.get_param(1)._data = bias;

    const auto status = linear.forward(input, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 8.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 0.5f);
}

TEST(linear_test, cpu_matrix_with_default_scale_and_no_bias) {
    const auto input = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    const auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    auto output = make_cpu_tensor({2, 2}, {0.f, 0.f, 0.f, 0.f});

    op::LinearOp linear;
    linear.reset_param_size(1);
    linear.get_param(0)._data = weight;

    const auto status = linear.forward(input, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 14.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 5.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[2], 32.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[3], 8.f);
}

void check_bf16_linear(base::DeviceType device_type, cudaStream_t stream = nullptr) {
    constexpr int K = 37;
    constexpr int M = 19;
    constexpr float scale = 0.7f;
    const std::array<uint16_t, 7> bits{0xc020, 0xbf00, 0x3c00, 0x3f81, 0x3d00, 0x4000, 0x0000};
    const std::array<float, 7> values{-2.5f, -0.5f, 0.0078125f, 1.0078125f, 0.03125f, 2.f, 0.f};
    std::vector<float> reference_values(M * K);
    tensor::Tensor bf16_weight(base::DataType::Bf16, {M, K},
                               base::CPUAllocatorFactory::get_instance());
    for (size_t index = 0; index < reference_values.size(); ++index) {
        bf16_weight.ptr<uint16_t>()[index] = bits[index % bits.size()];
        reference_values[index] = values[index % values.size()];
    }
    const auto reference_weight = make_cpu_tensor({M, K}, reference_values);
    if (device_type == base::DeviceType::GPU) {
        bf16_weight.to_cuda(stream);
    }

    for (int rows : {1, 17}) {
        SCOPED_TRACE(rows);
        const std::vector<int32_t> input_dims = rows == 1 ? std::vector<int32_t>{K}
                                                         : std::vector<int32_t>{rows, K};
        const std::vector<int32_t> output_dims = rows == 1 ? std::vector<int32_t>{M}
                                                          : std::vector<int32_t>{rows, M};
        std::vector<float> input_values(rows * K);
        for (size_t index = 0; index < input_values.size(); ++index) {
            // FP32 values that would change if activations were narrowed to BF16.
            input_values[index] = 0.01234567f * (static_cast<int>((index * 7) % 17) - 8);
        }
        auto input = make_cpu_tensor(input_dims, input_values);
        tensor::Tensor expected(base::DataType::Fp32, output_dims,
                                base::CPUAllocatorFactory::get_instance());
        std::vector<float> bias_values(expected.size());
        for (size_t index = 0; index < bias_values.size(); ++index) {
            bias_values[index] = 0.00321f * static_cast<float>(index % 11);
        }
        auto bias = make_cpu_tensor(output_dims, bias_values);
        op::LinearOp reference(scale);
        reference.reset_param_size(rows == 1 ? 1 : 2);
        reference.set_param(0, reference_weight);
        if (rows != 1) {
            reference.set_param(1, bias);
        }
        auto status = reference.forward(input, expected, cpu_context());
        ASSERT_TRUE(status.ok()) << status.message();

        op::OpContext context;
        context._device_type = device_type;
        context._stream = rows == 1 ? nullptr : stream;
        std::shared_ptr<base::DeviceAllocator> allocator = base::CPUAllocatorFactory::get_instance();
        if (device_type == base::DeviceType::GPU) {
            allocator = base::GPUAllocatorFactory::get_instance();
            input.to_cuda(context._stream);
            bias.to_cuda(context._stream);
        }
        tensor::Tensor output(base::DataType::Fp32, output_dims, allocator);
        op::LinearOp linear(scale);
        linear.reset_param_size(rows == 1 ? 1 : 2);
        linear.set_param(0, bf16_weight);
        if (rows != 1) {
            linear.set_param(1, bias);
        }
        status = linear.forward(input, output, context);
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_FALSE(linear.get_param(0).is_quantized());
        EXPECT_EQ(linear.get_param(0)._data.data_type(), base::DataType::Bf16);
        EXPECT_EQ(linear.get_param(0)._data.byte_size(), static_cast<size_t>(M * K * 2));
        if (device_type == base::DeviceType::GPU) {
            ASSERT_EQ(cudaGetLastError(), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(context._stream), cudaSuccess);
            output.to_cpu();
        }
        for (size_t index = 0; index < output.size(); ++index) {
            EXPECT_NEAR(output.ptr<float>()[index], expected.ptr<float>()[index], 1e-5f)
                << "output index " << index;
        }
    }
}

TEST(linear_test, cpu_bf16_weights_match_fp32_reference) {
    check_bf16_linear(base::DeviceType::CPU);
}

TEST(linear_test, bf16_weight_path_requires_fp32_activations_and_bias) {
    const auto allocator = base::CPUAllocatorFactory::get_instance();
    auto input = make_cpu_tensor({3}, {1.f, 2.f, 3.f});
    auto output = make_cpu_tensor({2}, {0.f, 0.f});
    tensor::Tensor weight(base::DataType::Bf16, {2, 3}, allocator);
    op::LinearOp linear;
    linear.reset_param_size(1);
    linear.set_param(0, weight);

    tensor::Tensor bf16_input(base::DataType::Bf16, {3}, allocator);
    EXPECT_EQ(linear.forward(bf16_input, output, cpu_context()).code(), base::InvalidArgument);
    tensor::Tensor bf16_output(base::DataType::Bf16, {2}, allocator);
    EXPECT_EQ(linear.forward(input, bf16_output, cpu_context()).code(), base::InvalidArgument);
    linear.set_param(0, tensor::Tensor(base::DataType::Bf16, {2, 4}, allocator));
    EXPECT_EQ(linear.forward(input, output, cpu_context()).code(), base::InvalidArgument);
    linear.set_param(0, tensor::Tensor(base::DataType::UInt8, {2, 3}, allocator));
    EXPECT_EQ(linear.forward(input, output, cpu_context()).code(), base::InvalidArgument);
    linear.set_param(0, weight);
    linear.reset_param_size(2);
    linear.set_param(1, tensor::Tensor(base::DataType::Bf16, {2}, allocator));
    EXPECT_EQ(linear.forward(input, output, cpu_context()).code(), base::InvalidArgument);
}

class LinearCudaTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int device_count = 0;
        const auto error = cudaGetDeviceCount(&device_count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
            (error == cudaSuccess && device_count == 0)) {
            GTEST_SKIP() << "CUDA unavailable";
        }
        ASSERT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    }
    void TearDown() override {
        if (stream) {
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    }
    cudaStream_t stream = nullptr;
};

TEST_F(LinearCudaTest, bf16_weights_match_fp32_reference_on_default_and_non_default_streams) {
    check_bf16_linear(base::DeviceType::GPU, stream);
}

TEST_F(LinearCudaTest, fp32_weights_preserve_scaled_matrix_results) {
    auto input = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto weight = make_cpu_tensor({2, 3}, {1.f, 2.f, 3.f, -1.f, 0.f, 2.f});
    input.to_cuda(stream);
    weight.to_cuda(stream);
    tensor::Tensor output(base::DataType::Fp32, {2, 2},
                          base::GPUAllocatorFactory::get_instance());
    op::LinearOp linear(0.5f);
    linear.reset_param_size(1);
    linear.set_param(0, weight);
    op::OpContext context;
    context._device_type = base::DeviceType::GPU;
    context._stream = stream;
    const auto status = linear.forward(input, output, context);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    output.to_cpu();
    const std::array<float, 4> expected{7.f, 2.5f, 16.f, 4.f};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_FLOAT_EQ(output.ptr<float>()[index], expected[index]);
    }
}

TEST(linear_test, cuda_quantized_int4_matches_cpu_fp32_matmul) {
    int device_count = 0;
    const auto device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(device_status);
    }

    constexpr int K = 256;
    constexpr int M = 19;
    constexpr int group_size = 128;
    constexpr int groups_per_row = K / group_size;

    std::vector<uint8_t> packed_weights(M * K / 2, 0);
    std::vector<float> fp32_weights(M * K);
    std::vector<float> scales(M * groups_per_row);
    std::vector<uint8_t> zero_points(M * groups_per_row);
    for (int m = 0; m < M; ++m) {
        for (int group = 0; group < groups_per_row; ++group) {
            const int index = m * groups_per_row + group;
            scales[index] = 0.125f * (1 + (m + group) % 3);
            zero_points[index] = static_cast<uint8_t>(4 + (m + 2 * group) % 8);
        }
        for (int k = 0; k < K; ++k) {
            const int group = k / group_size;
            const int index = m * groups_per_row + group;
            const uint8_t q = static_cast<uint8_t>((m * 7 + k * 3 + group * 5) % 16);
            auto& packed = packed_weights[m * (K / 2) + k / 2];
            packed |= static_cast<uint8_t>(q << ((k % 2) * 4));
            fp32_weights[m * K + k] =
                (static_cast<int>(q) - static_cast<int>(zero_points[index])) * scales[index];
        }
    }

    const auto reference_weight = make_cpu_tensor({M, K}, fp32_weights);
    op::LinearOp linear;
    linear.reset_param_size(1);
    auto& parameter = linear.get_param(0);
    parameter._data = tensor::Tensor(base::DataType::UInt8, {M, K / 2},
                                     base::CPUAllocatorFactory::get_instance());
    parameter._scales = make_cpu_tensor({M, groups_per_row}, scales);
    parameter._zero_points = tensor::Tensor(base::DataType::UInt8, {M, groups_per_row},
                                            base::CPUAllocatorFactory::get_instance());
    std::copy(packed_weights.begin(), packed_weights.end(), parameter._data.ptr<uint8_t>());
    std::copy(zero_points.begin(), zero_points.end(), parameter._zero_points.ptr<uint8_t>());
    parameter._quant_config._quant_type = op::QuantType::Int4GroupWise;
    parameter._quant_config._group_size = group_size;
    parameter._quant_config._symmetric = false;
    linear.to_cuda();

    for (int rows : {1, 17}) {
        SCOPED_TRACE(rows);
        std::vector<float> input_values(rows * K);
        for (int n = 0; n < rows; ++n) {
            for (int k = 0; k < K; ++k) {
                input_values[n * K + k] = 0.125f * ((n * 5 + k * 7) % 17 - 8);
            }
        }

        const std::vector<int32_t> input_dims = rows == 1 ? std::vector<int32_t>{K}
                                                           : std::vector<int32_t>{rows, K};
        const std::vector<int32_t> output_dims = rows == 1 ? std::vector<int32_t>{M}
                                                            : std::vector<int32_t>{rows, M};
        const auto cpu_input = make_cpu_tensor(input_dims, input_values);
        tensor::Tensor expected(base::DataType::Fp32, output_dims,
                                base::CPUAllocatorFactory::get_instance());
        op::MatmulOp matmul;
        const auto reference_status =
            matmul.forward(cpu_input, reference_weight, 1.0f, expected, cpu_context());
        ASSERT_TRUE(reference_status.ok()) << reference_status.message();

        auto gpu_input = cpu_input.clone();
        gpu_input.to_cuda();
        tensor::Tensor gpu_output(base::DataType::Fp32, output_dims,
                                  base::GPUAllocatorFactory::get_instance());
        op::OpContext context;
        context._device_type = base::DeviceType::GPU;
        cudaStream_t stream = nullptr;
        if (rows > 1) {
            const auto create_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            ASSERT_EQ(create_status, cudaSuccess) << cudaGetErrorString(create_status);
            context._stream = stream;
        }
        const auto status = linear.forward(gpu_input, gpu_output, context);
        ASSERT_TRUE(status.ok()) << status.message();
        const auto sync_status = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        ASSERT_EQ(sync_status, cudaSuccess) << cudaGetErrorString(sync_status);
        if (stream != nullptr) {
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
        gpu_output.to_cpu();

        for (size_t index = 0; index < expected.size(); ++index) {
            EXPECT_NEAR(gpu_output.ptr<float>()[index], expected.ptr<float>()[index], 1e-3f)
                << "output index " << index;
        }
    }
}

} // namespace
