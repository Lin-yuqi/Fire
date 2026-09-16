#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/rope.h"
#include "Fire/tensor/tensor.h"
#include "kernels_interface.h"

namespace {

constexpr float rope_theta = 10000.f;

struct RopeCase {
    int32_t head_size;
    int32_t cache_length;
    int32_t query_heads;
    int32_t key_heads;
};

// Include a partial position tile and more than 256 frequencies per head.
const RopeCase cases[] = {{2, 1, 1, 1}, {6, 7, 3, 1}, {64, 17, 4, 2},
                          {1026, 9, 3, 2}, {64, 2048, 4, 2}};

std::vector<float> input_values(int32_t heads, int32_t width, int32_t seed) {
    std::vector<float> values(static_cast<size_t>(heads) * width);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(static_cast<int32_t>((i * 7 + seed) % 29) - 14) *
                    0.125f;
    return values;
}

void expect_cache(const std::vector<float>& sin, const std::vector<float>& cos,
                  const RopeCase& shape) {
    const int32_t half = shape.head_size / 2;
    for (int32_t pos = 0; pos < shape.cache_length; ++pos) {
        // FP32 frequency rounding is amplified by the position multiplication.
        const double tolerance = 2e-6 + pos * 1e-7;
        for (int32_t frequency = 0; frequency < half; ++frequency) {
            const double angle = pos * std::pow(static_cast<double>(rope_theta),
                                                -2.0 * frequency / shape.head_size);
            const size_t index = static_cast<size_t>(pos) * half + frequency;
            ASSERT_NEAR(sin[index], std::sin(angle), tolerance)
                << "sin at position " << pos << ", frequency " << frequency;
            ASSERT_NEAR(cos[index], std::cos(angle), tolerance)
                << "cos at position " << pos << ", frequency " << frequency;
        }
    }
}

void expect_rotation(const std::vector<float>& original, const std::vector<float>& actual,
                     int32_t head_size, int32_t pos) {
    const int32_t half = head_size / 2;
    for (size_t i = 0; i < original.size(); ++i) {
        if (pos == 0) {
            ASSERT_FLOAT_EQ(actual[i], original[i]) << "index " << i;
            continue;
        }
        // Full-vector x*cos + rotate_half(x)*sin, independently evaluated in double.
        const int32_t column = static_cast<int32_t>(i % head_size);
        const double rotated = column < half ? -original[i + half] : original[i - half];
        const double angle = pos * std::pow(static_cast<double>(rope_theta),
                                            -2.0 * (column % half) / head_size);
        const double expected = original[i] * std::cos(angle) + rotated * std::sin(angle);
        ASSERT_NEAR(actual[i], expected, 1e-5 + pos * 4e-7) << "index " << i;
    }
}

void run_case(const RopeCase& shape, base::DeviceType device, cudaStream_t stream = nullptr) {
    SCOPED_TRACE(::testing::Message()
                 << "head size=" << shape.head_size << ", cache length=" << shape.cache_length);
    const bool gpu = device == base::DeviceType::GPU;
    std::shared_ptr<base::DeviceAllocator> allocator = base::CPUAllocatorFactory::get_instance();
    if (gpu)
        allocator = base::GPUAllocatorFactory::get_instance();
    tensor::Tensor query(base::DataType::Fp32, {shape.query_heads, shape.head_size}, allocator);
    tensor::Tensor key(base::DataType::Fp32, {shape.key_heads, shape.head_size}, allocator);
    tensor::Tensor sin(base::DataType::Fp32, {shape.cache_length, shape.head_size / 2}, allocator);
    tensor::Tensor cos(base::DataType::Fp32, {shape.cache_length, shape.head_size / 2}, allocator);
    const auto query_input = input_values(shape.query_heads, shape.head_size, 3);
    const auto key_input = input_values(shape.key_heads, shape.head_size, 11);
    std::vector<float> query_output(query.size()), key_output(key.size());
    std::vector<float> sin_output(sin.size()), cos_output(cos.size());
    op::OpContext context;
    context._device_type = device;
    context._stream = stream;
    op::RoPEOp rope;

    if (gpu) {
        ASSERT_EQ(cudaMemsetAsync(sin.ptr<float>(), 0xff, sin.byte_size(), stream), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(cos.ptr<float>(), 0xff, cos.byte_size(), stream), cudaSuccess);
    }
    kernel::get_rope_cache_kernel(device)(shape.head_size, shape.cache_length, rope_theta,
                                          sin, cos, stream);

    // The first nonzero rotation immediately consumes the cache on the same stream.
    for (const int32_t pos : {shape.cache_length - 1, 0}) {
        SCOPED_TRACE(pos);
        if (gpu) {
            ASSERT_EQ(cudaMemcpyAsync(query.ptr<float>(), query_input.data(), query.byte_size(),
                                      cudaMemcpyHostToDevice, stream), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(key.ptr<float>(), key_input.data(), key.byte_size(),
                                      cudaMemcpyHostToDevice, stream), cudaSuccess);
        } else {
            std::copy(query_input.begin(), query_input.end(), query.ptr<float>());
            std::copy(key_input.begin(), key_input.end(), key.ptr<float>());
        }

        const auto status = rope.forward(query, key, cos, sin, pos, context);
        ASSERT_TRUE(status.ok()) << status.message();
        if (gpu) {
            ASSERT_EQ(cudaMemcpyAsync(query_output.data(), query.ptr<float>(), query.byte_size(),
                                      cudaMemcpyDeviceToHost, stream), cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(key_output.data(), key.ptr<float>(), key.byte_size(),
                                      cudaMemcpyDeviceToHost, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        } else {
            std::copy_n(query.ptr<float>(), query.size(), query_output.begin());
            std::copy_n(key.ptr<float>(), key.size(), key_output.begin());
        }
        ASSERT_NO_FATAL_FAILURE(expect_rotation(query_input, query_output, shape.head_size, pos));
        ASSERT_NO_FATAL_FAILURE(expect_rotation(key_input, key_output, shape.head_size, pos));
    }

    if (gpu) {
        ASSERT_EQ(cudaMemcpyAsync(sin_output.data(), sin.ptr<float>(), sin.byte_size(),
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(cos_output.data(), cos.ptr<float>(), cos.byte_size(),
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    } else {
        std::copy_n(sin.ptr<float>(), sin.size(), sin_output.begin());
        std::copy_n(cos.ptr<float>(), cos.size(), cos_output.begin());
    }
    expect_cache(sin_output, cos_output, shape);
}

TEST(rope_test, cpu_compact_cache_and_half_split_gqa_match_formula) {
    for (const auto& shape : cases)
        ASSERT_NO_FATAL_FAILURE(run_case(shape, base::DeviceType::CPU));
}

TEST(rope_test, rejects_invalid_cache_shape_position_and_head_dimensions) {
    const auto allocator = base::CPUAllocatorFactory::get_instance();
    tensor::Tensor query(base::DataType::Fp32, {3, 6}, allocator);
    tensor::Tensor key(base::DataType::Fp32, {1, 6}, allocator);
    tensor::Tensor sin(base::DataType::Fp32, {7, 3}, allocator);
    tensor::Tensor cos(base::DataType::Fp32, {7, 3}, allocator);
    tensor::Tensor full_width(base::DataType::Fp32, {7, 6}, allocator);
    tensor::Tensor wrong_length(base::DataType::Fp32, {6, 3}, allocator);
    tensor::Tensor too_many_heads(base::DataType::Fp32, {4, 6}, allocator);
    tensor::Tensor odd_width(base::DataType::Fp32, {3, 5}, allocator);
    tensor::Tensor wrong_key_width(base::DataType::Fp32, {1, 8}, allocator);
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    op::RoPEOp rope;

    EXPECT_EQ(rope.forward(query, key, full_width, full_width, 1, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(rope.forward(query, key, cos, wrong_length, 1, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(rope.forward(query, key, cos, sin, -1, context).code(), base::InvalidArgument);
    EXPECT_EQ(rope.forward(query, key, cos, sin, 7, context).code(), base::InvalidArgument);
    EXPECT_EQ(rope.forward(query, too_many_heads, cos, sin, 1, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(rope.forward(odd_width, key, cos, sin, 1, context).code(), base::InvalidArgument);
    EXPECT_EQ(rope.forward(query, wrong_key_width, cos, sin, 1, context).code(),
              base::InvalidArgument);
}

class RopeCudaTest : public ::testing::Test {
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

TEST_F(RopeCudaTest, compact_cache_and_half_split_gqa_on_non_default_stream) {
    for (const auto& shape : cases)
        ASSERT_NO_FATAL_FAILURE(run_case(shape, base::DeviceType::GPU, stream));
}

} // namespace
