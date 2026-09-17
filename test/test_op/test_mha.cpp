#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/mha.h"
#include "Fire/tensor/tensor.h"
#include "cpu/mha_kernel.h"
#include "cuda/mha_kernel.cuh"
#include "kernels_interface.h"

namespace {

constexpr float score_sentinel = -777.0f;

struct MhaShape {
    int32_t layers;
    int32_t capacity;
    int32_t query_heads;
    int32_t kv_heads;
    int32_t head_dim;
};

struct MhaReference {
    std::vector<double> score;
    std::vector<double> output;
};

tensor::Tensor cpu_tensor(std::vector<int32_t> dims,
                          base::DataType dtype = base::DataType::Fp32) {
    return tensor::Tensor(dtype, std::move(dims), base::CPUAllocatorFactory::get_instance());
}

op::OpContext cpu_context() {
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    return context;
}

size_t cache_index(const MhaShape& shape, int32_t layer, int32_t token, int32_t head,
                   int32_t column) {
    return (((static_cast<size_t>(layer) * shape.capacity + token) * shape.kv_heads + head) *
            shape.head_dim) +
           column;
}

std::vector<float> make_values(size_t size, int32_t multiplier, int32_t offset, float scale) {
    std::vector<float> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] =
            static_cast<float>(static_cast<int32_t>((i * multiplier + offset) % 31) - 15) * scale;
    }
    return values;
}

MhaReference reference_mha(const MhaShape& shape, const std::vector<float>& query,
                           const std::vector<float>& key, const std::vector<float>& value,
                           int32_t layer, int32_t pos) {
    MhaReference reference{
        std::vector<double>(static_cast<size_t>(shape.query_heads) * shape.capacity,
                            score_sentinel),
        std::vector<double>(static_cast<size_t>(shape.query_heads) * shape.head_dim)};
    const int32_t kv_group_size = shape.query_heads / shape.kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(shape.head_dim));

    for (int32_t query_head = 0; query_head < shape.query_heads; ++query_head) {
        const int32_t kv_head = query_head / kv_group_size;
        std::vector<double> logits(pos + 1);
        for (int32_t token = 0; token <= pos; ++token) {
            double dot = 0.0;
            for (int32_t column = 0; column < shape.head_dim; ++column) {
                dot += query[static_cast<size_t>(query_head) * shape.head_dim + column] *
                       key[cache_index(shape, layer, token, kv_head, column)];
            }
            logits[token] = dot * scale;
        }

        const double maximum = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (double& logit : logits) {
            logit = std::exp(logit - maximum);
            sum += logit;
        }

        for (int32_t token = 0; token <= pos; ++token) {
            const double weight = logits[token] / sum;
            reference.score[static_cast<size_t>(query_head) * shape.capacity + token] = weight;
            for (int32_t column = 0; column < shape.head_dim; ++column) {
                reference.output[static_cast<size_t>(query_head) * shape.head_dim + column] +=
                    weight * value[cache_index(shape, layer, token, kv_head, column)];
            }
        }
    }

    return reference;
}

void run_numeric_case(const MhaShape& shape, int32_t layer, int32_t pos,
                      base::DeviceType device, cudaStream_t stream = nullptr) {
    SCOPED_TRACE(::testing::Message() << "layers=" << shape.layers
                                     << ", capacity=" << shape.capacity
                                     << ", query heads=" << shape.query_heads
                                     << ", KV heads=" << shape.kv_heads
                                     << ", head dim=" << shape.head_dim << ", layer=" << layer
                                     << ", pos=" << pos);
    const bool gpu = device == base::DeviceType::GPU;
    std::shared_ptr<base::DeviceAllocator> allocator = base::CPUAllocatorFactory::get_instance();
    if (gpu) {
        allocator = base::GPUAllocatorFactory::get_instance();
    }

    tensor::Tensor query(base::DataType::Fp32, {shape.query_heads, shape.head_dim}, allocator);
    tensor::Tensor key_cache(base::DataType::Fp32,
                             {shape.layers, shape.capacity, shape.kv_heads, shape.head_dim},
                             allocator);
    tensor::Tensor value_cache(base::DataType::Fp32,
                               {shape.layers, shape.capacity, shape.kv_heads, shape.head_dim},
                               allocator);
    tensor::Tensor score(base::DataType::Fp32, {shape.query_heads, shape.capacity}, allocator);
    tensor::Tensor output(base::DataType::Fp32, {shape.query_heads, shape.head_dim}, allocator);

    const auto query_values = make_values(query.size(), 7, 3, 0.125f);
    const auto key_values = make_values(key_cache.size(), 11, 5, 0.0625f);
    const auto value_values = make_values(value_cache.size(), 13, 9, 0.1875f);
    const std::vector<float> initial_score(score.size(), score_sentinel);
    const auto reference =
        reference_mha(shape, query_values, key_values, value_values, layer, pos);
    std::vector<float> actual_score(score.size());
    std::vector<float> actual_output(output.size());
    std::vector<float> actual_query(query.size());
    std::vector<float> actual_key(key_cache.size());
    std::vector<float> actual_value(value_cache.size());

    if (gpu) {
        ASSERT_EQ(cudaMemcpyAsync(query.ptr<float>(), query_values.data(), query.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(key_cache.ptr<float>(), key_values.data(), key_cache.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(value_cache.ptr<float>(), value_values.data(),
                                  value_cache.byte_size(), cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(score.ptr<float>(), initial_score.data(), score.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
    } else {
        std::copy(query_values.begin(), query_values.end(), query.ptr<float>());
        std::copy(key_values.begin(), key_values.end(), key_cache.ptr<float>());
        std::copy(value_values.begin(), value_values.end(), value_cache.ptr<float>());
        std::copy(initial_score.begin(), initial_score.end(), score.ptr<float>());
    }

    op::MultiHeadAttentionOp mha;
    op::OpContext context;
    context._device_type = device;
    context._stream = stream;
    const auto status =
        mha.forward(query, key_cache, value_cache, score, output, layer, pos, context);

    ASSERT_TRUE(status.ok()) << status.message();
    if (gpu) {
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual_score.data(), score.ptr<float>(), score.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual_output.data(), output.ptr<float>(), output.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual_query.data(), query.ptr<float>(), query.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual_key.data(), key_cache.ptr<float>(), key_cache.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(actual_value.data(), value_cache.ptr<float>(),
                                  value_cache.byte_size(), cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    } else {
        std::copy_n(score.ptr<float>(), score.size(), actual_score.begin());
        std::copy_n(output.ptr<float>(), output.size(), actual_output.begin());
        std::copy_n(query.ptr<float>(), query.size(), actual_query.begin());
        std::copy_n(key_cache.ptr<float>(), key_cache.size(), actual_key.begin());
        std::copy_n(value_cache.ptr<float>(), value_cache.size(), actual_value.begin());
    }

    EXPECT_EQ(actual_query, query_values);
    EXPECT_EQ(actual_key, key_values);
    EXPECT_EQ(actual_value, value_values);
    for (int32_t head = 0; head < shape.query_heads; ++head) {
        double score_sum = 0.0;
        for (int32_t token = 0; token < shape.capacity; ++token) {
            const size_t index = static_cast<size_t>(head) * shape.capacity + token;
            if (token <= pos) {
                EXPECT_NEAR(actual_score[index], reference.score[index], 2e-5)
                    << "score head=" << head << ", token=" << token;
                score_sum += actual_score[index];
            } else {
                EXPECT_FLOAT_EQ(actual_score[index], score_sentinel)
                    << "score tail head=" << head << ", token=" << token;
            }
        }
        EXPECT_NEAR(score_sum, 1.0, 3e-5) << "head=" << head;
    }
    for (size_t i = 0; i < actual_output.size(); ++i) {
        EXPECT_NEAR(actual_output[i], reference.output[i], 3e-5) << "output index=" << i;
    }
}

TEST(mha_kernel_interface_test, selects_kernel_for_each_supported_device) {
    EXPECT_EQ(kernel::get_mha_kernel(base::DeviceType::CPU), kernel::mha_kernel_cpu);
    EXPECT_EQ(kernel::get_mha_kernel(base::DeviceType::GPU), kernel::mha_kernel_cu);
}

TEST(mha_test, cpu_forward_matches_gqa_reference_and_preserves_inputs) {
    ASSERT_NO_FATAL_FAILURE(
        run_numeric_case({2, 5, 4, 2, 3}, 1, 3, base::DeviceType::CPU));
}

TEST(mha_test, rejects_invalid_tensors_shapes_indices_and_context) {
    auto query = cpu_tensor({4, 3});
    auto key_cache = cpu_tensor({2, 5, 2, 3});
    auto value_cache = cpu_tensor({2, 5, 2, 3});
    auto score = cpu_tensor({4, 5});
    auto output = cpu_tensor({4, 3});
    auto context = cpu_context();
    op::MultiHeadAttentionOp mha;

    EXPECT_EQ(mha.type(), op::OpType::MHA);
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, -1, 0, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, 2, 0, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, 0, -1, context).code(),
              base::InvalidArgument);
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, 0, 5, context).code(),
              base::InvalidArgument);

    op::OpContext unknown_context;
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, 0, 0, unknown_context)
                  .code(),
              base::InvalidArgument);
    auto gpu_context = context;
    gpu_context._device_type = base::DeviceType::GPU;
    EXPECT_EQ(mha.forward(query, key_cache, value_cache, score, output, 0, 0, gpu_context).code(),
              base::InvalidArgument);

    for (int position = 0; position < 5; ++position) {
        SCOPED_TRACE(::testing::Message() << "wrong dtype at tensor position=" << position);
        std::vector<tensor::Tensor> tensors = {cpu_tensor({4, 3}), cpu_tensor({2, 5, 2, 3}),
                                                cpu_tensor({2, 5, 2, 3}), cpu_tensor({4, 5}),
                                                cpu_tensor({4, 3})};
        const auto dims = tensors[position].dims();
        tensors[position] = cpu_tensor(dims, base::DataType::int32);
        EXPECT_EQ(mha.forward(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], 0, 0,
                              context)
                      .code(),
                  base::InvalidArgument);
    }

    auto empty_query = tensor::Tensor{};
    EXPECT_EQ(mha.forward(empty_query, key_cache, value_cache, score, output, 0, 0, context).code(),
              base::InvalidArgument);
    auto unallocated_query = tensor::Tensor(base::DataType::Fp32, {4, 3});
    EXPECT_EQ(
        mha.forward(unallocated_query, key_cache, value_cache, score, output, 0, 0, context).code(),
        base::InvalidArgument);

    auto flat_query = cpu_tensor({12});
    EXPECT_EQ(mha.forward(flat_query, key_cache, value_cache, score, output, 0, 0, context).code(),
              base::InvalidArgument);
    auto flat_key_cache = cpu_tensor({60});
    EXPECT_EQ(mha.forward(query, flat_key_cache, value_cache, score, output, 0, 0, context).code(),
              base::InvalidArgument);
    auto wrong_value_shape = cpu_tensor({2, 4, 2, 3});
    EXPECT_EQ(
        mha.forward(query, key_cache, wrong_value_shape, score, output, 0, 0, context).code(),
        base::InvalidArgument);
    auto wrong_head_dim_cache = cpu_tensor({2, 5, 2, 4});
    EXPECT_EQ(mha.forward(query, wrong_head_dim_cache, wrong_head_dim_cache, score, output, 0, 0,
                          context)
                  .code(),
              base::InvalidArgument);

    auto three_head_query = cpu_tensor({3, 3});
    auto three_head_score = cpu_tensor({3, 5});
    auto three_head_output = cpu_tensor({3, 3});
    EXPECT_EQ(mha.forward(three_head_query, key_cache, value_cache, three_head_score,
                          three_head_output, 0, 0, context)
                  .code(),
              base::InvalidArgument);

    auto wrong_score = cpu_tensor({4, 4});
    EXPECT_EQ(
        mha.forward(query, key_cache, value_cache, wrong_score, output, 0, 0, context).code(),
        base::InvalidArgument);
    auto flat_output = cpu_tensor({12});
    EXPECT_EQ(
        mha.forward(query, key_cache, value_cache, score, flat_output, 0, 0, context).code(),
        base::InvalidArgument);
}

class MhaCudaTest : public ::testing::Test {
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

TEST_F(MhaCudaTest, forward_matches_reference_past_one_block_on_non_default_stream) {
    ASSERT_NO_FATAL_FAILURE(
        run_numeric_case({2, 257, 4, 2, 5}, 1, 256, base::DeviceType::GPU, stream));
}

} // namespace
