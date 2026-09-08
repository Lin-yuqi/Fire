#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/tensor/tensor.h"

namespace {

std::vector<float> rmsnorm_reference(const std::vector<float>& input,
                                     const std::vector<float>& weight, int32_t width,
                                     float eps) {
    std::vector<float> result(input.size());
    const int32_t rows = static_cast<int32_t>(input.size()) / width;
    for (int32_t row = 0; row < rows; ++row) {
        const int32_t offset = row * width;
        float square_sum = 0.f;
        for (int32_t col = 0; col < width; ++col) {
            const float value = input[offset + col];
            square_sum += value * value;
        }
        const float scale = 1.f / std::sqrt(square_sum / static_cast<float>(width) + eps);
        for (int32_t col = 0; col < width; ++col)
            result[offset + col] = input[offset + col] * weight[col] * scale;
    }
    return result;
}

void fill_inputs(std::vector<float>& input, std::vector<float>& weight) {
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int32_t>(i * 7 % 29) - 14) * 0.125f;
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = 0.75f + static_cast<float>(i % 11) * 0.03125f;
}

tensor::Tensor cpu_tensor(std::vector<int32_t> dims,
                          base::DataType dtype = base::DataType::Fp32) {
    return tensor::Tensor(dtype, std::move(dims), base::CPUAllocatorFactory::get_instance());
}

TEST(rmsnorm_test, cpu_fp32_correctness_and_custom_epsilon) {
    constexpr int32_t size = 513;
    constexpr float eps = 1e-3f;
    std::vector<float> input_values(size);
    std::vector<float> weight_values(size);
    fill_inputs(input_values, weight_values);
    const auto expected = rmsnorm_reference(input_values, weight_values, size, eps);

    auto input = cpu_tensor({size});
    auto weight = cpu_tensor({size});
    auto output = cpu_tensor({size});
    std::copy(input_values.begin(), input_values.end(), input.ptr<float>());
    std::copy(weight_values.begin(), weight_values.end(), weight.ptr<float>());

    op::RmsNormOp rmsnorm(eps);
    rmsnorm.reset_param_size(1);
    rmsnorm.get_param(0)._data = weight;
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;

    const auto status = rmsnorm.forward(input, output, context);
    ASSERT_TRUE(status.ok()) << status.message();
    for (int32_t i = 0; i < size; ++i) {
        EXPECT_NEAR(output.ptr<float>()[i], expected[i], 1e-5f) << "index = " << i;
        EXPECT_FLOAT_EQ(input.ptr<float>()[i], input_values[i]);
        EXPECT_FLOAT_EQ(weight.ptr<float>()[i], weight_values[i]);
    }
}

TEST(rmsnorm_test, rejects_invalid_parameters_and_tensors) {
    auto input = cpu_tensor({8});
    auto output = cpu_tensor({8});
    auto weight = cpu_tensor({8});
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    op::RmsNormOp rmsnorm;

    EXPECT_EQ(rmsnorm.forward(input, output, context).code(), base::InvalidArgument);

    rmsnorm.reset_param_size(2);
    EXPECT_EQ(rmsnorm.forward(input, output, context).code(), base::InvalidArgument);

    rmsnorm.reset_param_size(1);
    rmsnorm.get_param(0)._data = weight;
    rmsnorm.get_param(0)._quant_config._quant_type = op::QuantType::Int8PerTensor;
    EXPECT_EQ(rmsnorm.forward(input, output, context).code(), base::InvalidArgument);

    rmsnorm.get_param(0)._quant_config._quant_type = op::QuantType::None;
    rmsnorm.get_param(0)._data = cpu_tensor({7});
    EXPECT_EQ(rmsnorm.forward(input, output, context).code(), base::InvalidArgument);

    rmsnorm.get_param(0)._data = weight;
    auto wrong_shape_output = cpu_tensor({4, 2});
    EXPECT_EQ(rmsnorm.forward(input, wrong_shape_output, context).code(), base::InvalidArgument);

    auto wrong_type_output = cpu_tensor({8}, base::DataType::int32);
    EXPECT_EQ(rmsnorm.forward(input, wrong_type_output, context).code(), base::InvalidArgument);

    context._device_type = base::DeviceType::Unknown;
    EXPECT_EQ(rmsnorm.forward(input, output, context).code(), base::InvalidArgument);
}

class RmsNormCudaTest : public ::testing::Test {
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
        if (stream)
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    void run_case(const std::vector<int32_t>& dims, float eps) {
        const int32_t width = dims.back();
        int32_t total_size = 1;
        for (const int32_t dim : dims)
            total_size *= dim;

        std::vector<float> input_values(total_size);
        std::vector<float> weight_values(width);
        std::vector<float> output_values(total_size);
        fill_inputs(input_values, weight_values);
        const auto expected = rmsnorm_reference(input_values, weight_values, width, eps);

        auto allocator = base::GPUAllocatorFactory::get_instance();
        tensor::Tensor input(base::DataType::Fp32, dims, allocator);
        tensor::Tensor weight(base::DataType::Fp32, {width}, allocator);
        tensor::Tensor output(base::DataType::Fp32, dims, allocator);
        ASSERT_EQ(cudaMemcpyAsync(input.ptr<float>(), input_values.data(), input.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(weight.ptr<float>(), weight_values.data(), weight.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);

        op::RmsNormOp rmsnorm(eps);
        rmsnorm.reset_param_size(1);
        rmsnorm.get_param(0)._data = weight;
        op::OpContext context;
        context._device_type = base::DeviceType::GPU;
        context._stream = stream;

        const auto status = rmsnorm.forward(input, output, context);
        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(output_values.data(), output.ptr<float>(), output.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (int32_t i = 0; i < total_size; ++i)
            EXPECT_NEAR(output_values[i], expected[i], 1e-5f) << "index = " << i;
    }

    cudaStream_t stream = nullptr;
};

TEST_F(RmsNormCudaTest, fp32_vector_sizes_cover_packed_tail) {
    for (const int32_t size : {1, 3, 4, 255, 256, 257, 1025, 4096}) {
        SCOPED_TRACE(size);
        run_case({size}, 1e-6f);
    }
}

TEST_F(RmsNormCudaTest, fp32_rows_with_width_not_divisible_by_four) {
    run_case({3, 1025}, 1e-4f);
}

} // namespace
