#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/tensor/tensor.h"
#include "cpu/softmax_kernel.h"
#include "cuda/softmax_kernel.cuh"

namespace {

std::vector<double> softmax_reference(const std::vector<float>& input, int32_t width) {
    std::vector<double> result(input.size());
    for (size_t offset = 0; offset < input.size(); offset += width) {
        const double maximum = *std::max_element(input.begin() + offset,
                                                 input.begin() + offset + width);
        double sum = 0.0;
        for (int32_t column = 0; column < width; ++column) {
            result[offset + column] = std::exp(static_cast<double>(input[offset + column]) -
                                               maximum);
            sum += result[offset + column];
        }
        for (int32_t column = 0; column < width; ++column)
            result[offset + column] /= sum;
    }
    return result;
}

void run_case(const std::vector<int32_t>& dims, const std::vector<float>& values,
              base::DeviceType device, bool in_place, cudaStream_t stream) {
    SCOPED_TRACE(::testing::Message() << "rank=" << dims.size() << ", rows="
                                     << values.size() / dims.back() << ", width=" << dims.back()
                                     << ", in_place=" << in_place);
    const bool gpu = device == base::DeviceType::GPU;
    std::shared_ptr<base::DeviceAllocator> allocator = base::CPUAllocatorFactory::get_instance();
    if (gpu)
        allocator = base::GPUAllocatorFactory::get_instance();
    tensor::Tensor input(base::DataType::Fp32, dims, allocator);
    tensor::Tensor output = in_place ? input : tensor::Tensor(base::DataType::Fp32, dims, allocator);
    const auto expected = softmax_reference(values, dims.back());
    std::vector<float> actual(values.size()), preserved(values.size());

    if (gpu) {
        ASSERT_EQ(cudaMemcpyAsync(input.ptr<float>(), values.data(), input.byte_size(),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        if (!in_place)
            ASSERT_EQ(cudaMemsetAsync(output.ptr<float>(), 0xff, output.byte_size(), stream),
                      cudaSuccess);
    } else {
        std::copy(values.begin(), values.end(), input.ptr<float>());
        if (!in_place)
            std::fill_n(output.ptr<float>(), output.size(),
                        std::numeric_limits<float>::quiet_NaN());
    }

    if (gpu)
        kernel::softmax_kernel_cu(input, output, stream);
    else
        kernel::softmax_kernel_cpu(input, output, stream);

    if (gpu) {
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), output.ptr<float>(), output.byte_size(),
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        if (!in_place)
            ASSERT_EQ(cudaMemcpyAsync(preserved.data(), input.ptr<float>(), input.byte_size(),
                                      cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    } else {
        std::copy_n(output.ptr<float>(), output.size(), actual.begin());
        if (!in_place)
            std::copy_n(input.ptr<float>(), input.size(), preserved.begin());
    }

    if (!in_place)
        ASSERT_EQ(preserved, values) << "softmax modified its input";
    for (size_t offset = 0; offset < values.size(); offset += dims.back()) {
        double sum = 0.0;
        for (int32_t column = 0; column < dims.back(); ++column) {
            const size_t index = offset + column;
            ASSERT_TRUE(std::isfinite(actual[index])) << "index=" << index;
            ASSERT_GE(actual[index], 0.0f) << "index=" << index;
            ASSERT_LE(actual[index], 1.0f) << "index=" << index;
            ASSERT_NEAR(actual[index], expected[index], 1e-7 + 2e-5 * expected[index])
                << "index=" << index;
            if (values[index] == -std::numeric_limits<float>::infinity())
                ASSERT_EQ(actual[index], 0.0f) << "masked index=" << index;
            sum += actual[index];
        }
        ASSERT_NEAR(sum, 1.0, 2e-5) << "row=" << offset / dims.back();
    }
}

void run_boundary_cases(base::DeviceType device, cudaStream_t stream = nullptr) {
    for (const int32_t width : {1, 31, 32, 33, 255, 256, 257, 511, 512, 513,
                                 1023, 1024, 1025, 2047, 2048, 2049, 4097}) {
        for (const std::vector<int32_t>& dims :
             {std::vector<int32_t>{width}, {1, width}, {3, width}}) {
            const int32_t rows = dims.size() == 1 ? 1 : dims.front();
            std::vector<float> values(static_cast<size_t>(rows) * width);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(static_cast<int32_t>((i * 7) % 101) - 50) *
                                0.125f +
                            static_cast<float>(i / width) * 1000.0f;
            for (const bool in_place : {false, true})
                ASSERT_NO_FATAL_FAILURE(run_case(dims, values, device, in_place, stream));
        }
    }
}

void run_numeric_cases(base::DeviceType device, cudaStream_t stream = nullptr) {
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    // Large shifts, equal logits, and masked entries with at least one finite logit per row.
    const std::vector<float> values = {-1000.0f, -1001.0f, -1002.0f, -1003.0f,
                                        1000.0f, -1000.0f, 0.0f, 1000.0f,
                                        -1234.0f, -1234.0f, -1234.0f, -1234.0f,
                                        negative_infinity, 0.0f, negative_infinity, 1.0f};
    for (const bool in_place : {false, true}) {
        ASSERT_NO_FATAL_FAILURE(
            run_case({2}, {-1000.0f, -1001.0f}, device, in_place, stream));
        ASSERT_NO_FATAL_FAILURE(run_case({4, 4}, values, device, in_place, stream));
    }
}

TEST(softmax_test, cpu_matches_reference_at_width_boundaries) {
    run_boundary_cases(base::DeviceType::CPU);
}

TEST(softmax_test, cpu_handles_large_equal_and_masked_logits) {
    run_numeric_cases(base::DeviceType::CPU);
}

class SoftmaxCudaTest : public ::testing::Test {
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

    cudaStream_t stream = nullptr;
};

TEST_F(SoftmaxCudaTest, matches_reference_at_width_boundaries_on_non_default_stream) {
    run_boundary_cases(base::DeviceType::GPU, stream);
}

TEST_F(SoftmaxCudaTest, handles_large_equal_and_masked_logits) {
    run_numeric_cases(base::DeviceType::GPU, stream);
}

} // namespace
