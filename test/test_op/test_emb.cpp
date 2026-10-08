#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/embedding.h"
#include "Fire/tensor/tensor.h"
#include "cpu/emb_kernel.h"
#include "cuda/emb_kernel.cuh"
#include "kernels_interface.h"

namespace {

tensor::Tensor cpu_tensor(std::vector<int32_t> dims,
                          base::DataType dtype = base::DataType::Fp32) {
    return tensor::Tensor(dtype, std::move(dims),
                          base::CPUAllocatorFactory::get_instance());
}

std::vector<float> make_weight_values(int32_t vocab_size, int32_t embedding_dim) {
    std::vector<float> values(static_cast<size_t>(vocab_size) * embedding_dim);
    for (int32_t token = 0; token < vocab_size; ++token) {
        for (int32_t col = 0; col < embedding_dim; ++col) {
            values[static_cast<size_t>(token) * embedding_dim + col] =
                static_cast<float>(token * 10 + col) + 0.25f;
        }
    }
    return values;
}

void expect_selected_rows(const std::vector<int32_t>& tokens,
                          const std::vector<float>& weight, int32_t embedding_dim,
                          const float* output) {
    for (size_t row = 0; row < tokens.size(); ++row) {
        for (int32_t col = 0; col < embedding_dim; ++col) {
            const size_t output_index = row * embedding_dim + col;
            const size_t weight_index =
                static_cast<size_t>(tokens[row]) * embedding_dim + col;
            EXPECT_FLOAT_EQ(output[output_index], weight[weight_index])
                << "row = " << row << ", col = " << col;
        }
    }
}

op::OpContext cpu_context() {
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    return context;
}

void check_bf16_embedding(base::DeviceType device_type, cudaStream_t stream = nullptr) {
    constexpr int32_t vocab_size = 6;
    const std::array<uint16_t, 8> bits{
        0x0000, 0x8000, 0x3f80, 0xc020, 0x3f81, 0x3380, 0x7180, 0xbf00};
    const std::array<float, 8> values{
        0.f, -0.f, 1.f, -2.5f, 1.0078125f, 0x1p-24f, 0x1p100f, -0.5f};
    std::shared_ptr<base::DeviceAllocator> output_allocator =
        base::CPUAllocatorFactory::get_instance();
    if (device_type == base::DeviceType::GPU) {
        output_allocator = base::GPUAllocatorFactory::get_instance();
    }
    for (int32_t dim : {1, 5, 257}) {
        for (const auto& token_values :
             {std::vector<int32_t>{4}, std::vector<int32_t>{3, 0, 5, 3}}) {
            SCOPED_TRACE(::testing::Message() << "dim=" << dim
                                              << " tokens=" << token_values.size());
            const auto token_count = static_cast<int32_t>(token_values.size());
            auto tokens = cpu_tensor({token_count}, base::DataType::Int32);
            auto weight = cpu_tensor({vocab_size, dim}, base::DataType::Bf16);
            std::vector<float> reference(weight.size());
            for (size_t index = 0; index < weight.size(); ++index) {
                weight.ptr<uint16_t>()[index] = bits[index % bits.size()];
                reference[index] = values[index % values.size()];
            }
            std::copy(token_values.begin(), token_values.end(), tokens.ptr<int32_t>());
            if (device_type == base::DeviceType::GPU) {
                tokens.to_cuda(stream);
                weight.to_cuda(stream);
            }
            const std::vector<int32_t> output_dims = token_count == 1
                                                        ? std::vector<int32_t>{dim}
                                                        : std::vector<int32_t>{token_count, dim};
            tensor::Tensor output(base::DataType::Fp32, output_dims, output_allocator);
            op::EmbeddingOp embedding;
            embedding.reset_param_size(1);
            embedding.set_param(0, weight);
            op::OpContext context;
            context._device_type = device_type;
            context._stream = stream;
            const auto status = embedding.forward(tokens, output, context);
            ASSERT_TRUE(status.ok()) << status.message();
            EXPECT_EQ(embedding.get_param(0)._data.data_type(), base::DataType::Bf16);
            EXPECT_EQ(embedding.get_param(0)._data.byte_size(), weight.size() * 2);

            std::vector<float> observed(output.size());
            if (device_type == base::DeviceType::GPU) {
                ASSERT_EQ(cudaGetLastError(), cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(observed.data(), output.ptr<float>(), output.byte_size(),
                                          cudaMemcpyDeviceToHost, stream), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            } else {
                std::copy_n(output.ptr<float>(), output.size(), observed.begin());
            }
            expect_selected_rows(token_values, reference, dim, observed.data());
            for (size_t row = 0; row < token_values.size(); ++row) {
                for (int32_t col = 0; col < dim; ++col) {
                    const size_t weight_index = static_cast<size_t>(token_values[row]) * dim + col;
                    EXPECT_EQ(std::signbit(observed[row * dim + col]),
                              std::signbit(reference[weight_index]));
                }
            }
        }
    }
}

TEST(embedding_kernel_interface_test, selects_kernel_for_each_supported_device) {
    EXPECT_EQ(kernel::get_embedding_kernel(base::DeviceType::CPU), kernel::emb_kernel_cpu);
    EXPECT_EQ(kernel::get_embedding_kernel(base::DeviceType::GPU), kernel::emb_kernel_cu);
}

TEST(embedding_kernel_interface_death_test, rejects_unknown_device) {
    EXPECT_DEATH(
        {
            const auto selected =
                kernel::get_embedding_kernel(base::DeviceType::Unknown);
            (void)selected;
        },
        "Unknown device type");
}

TEST(embedding_test, cpu_forward_selects_rows_in_token_order) {
    ASSERT_EQ(kernel::get_embedding_kernel(base::DeviceType::CPU), kernel::emb_kernel_cpu)
        << "the CPU embedding kernel selector must be implemented before forward can run";

    constexpr int32_t vocab_size = 5;
    constexpr int32_t embedding_dim = 3;
    const std::vector<int32_t> token_values = {3, 0, 4};
    const auto weight_values = make_weight_values(vocab_size, embedding_dim);

    auto tokens = cpu_tensor({static_cast<int32_t>(token_values.size())},
                             base::DataType::Int32);
    auto weight = cpu_tensor({vocab_size, embedding_dim});
    auto output = cpu_tensor({static_cast<int32_t>(token_values.size()), embedding_dim});
    std::copy(token_values.begin(), token_values.end(), tokens.ptr<int32_t>());
    std::copy(weight_values.begin(), weight_values.end(), weight.ptr<float>());

    op::EmbeddingOp embedding;
    embedding.reset_param_size(1);
    embedding.set_param(0, weight);

    const auto status = embedding.forward(tokens, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    expect_selected_rows(token_values, weight_values, embedding_dim, output.ptr<float>());
}

TEST(embedding_test, cpu_forward_accepts_one_dimensional_single_token_output) {
    ASSERT_EQ(kernel::get_embedding_kernel(base::DeviceType::CPU), kernel::emb_kernel_cpu)
        << "the CPU embedding kernel selector must be implemented before forward can run";

    constexpr int32_t vocab_size = 4;
    constexpr int32_t embedding_dim = 5;
    const std::vector<int32_t> token_values = {2};
    const auto weight_values = make_weight_values(vocab_size, embedding_dim);

    auto tokens = cpu_tensor({1}, base::DataType::Int32);
    auto weight = cpu_tensor({vocab_size, embedding_dim});
    auto output = cpu_tensor({embedding_dim});
    tokens.ptr<int32_t>()[0] = token_values[0];
    std::copy(weight_values.begin(), weight_values.end(), weight.ptr<float>());

    op::EmbeddingOp embedding;
    embedding.reset_param_size(1);
    embedding.set_param(0, weight);

    const auto status = embedding.forward(tokens, output, cpu_context());

    ASSERT_TRUE(status.ok()) << status.message();
    expect_selected_rows(token_values, weight_values, embedding_dim, output.ptr<float>());
}

TEST(embedding_test, cpu_bf16_weights_produce_fp32_embeddings) {
    check_bf16_embedding(base::DeviceType::CPU);
}

TEST(embedding_test, rejects_invalid_parameters_and_tensors) {
    auto tokens = cpu_tensor({2}, base::DataType::Int32);
    auto weight = cpu_tensor({4, 3});
    auto output = cpu_tensor({2, 3});
    op::EmbeddingOp embedding;

    EXPECT_EQ(embedding.forward(tokens, output, cpu_context()).code(), base::InvalidArgument);

    embedding.reset_param_size(2);
    EXPECT_EQ(embedding.forward(tokens, output, cpu_context()).code(), base::InvalidArgument);

    embedding.reset_param_size(1);
    embedding.set_param(0, weight);
    embedding.get_param(0)._quant_config._quant_type = op::QuantType::Int8PerTensor;
    EXPECT_EQ(embedding.forward(tokens, output, cpu_context()).code(), base::InvalidArgument);

    embedding.get_param(0)._quant_config._quant_type = op::QuantType::None;
    auto wrong_token_type = cpu_tensor({2}, base::DataType::Fp32);
    EXPECT_EQ(embedding.forward(wrong_token_type, output, cpu_context()).code(),
              base::InvalidArgument);

    auto wrong_output_type = cpu_tensor({2, 3}, base::DataType::Int32);
    EXPECT_EQ(embedding.forward(tokens, wrong_output_type, cpu_context()).code(),
              base::InvalidArgument);

    embedding.set_param(0, cpu_tensor({4, 3}, base::DataType::Int32));
    EXPECT_EQ(embedding.forward(tokens, output, cpu_context()).code(), base::InvalidArgument);

    embedding.set_param(0, cpu_tensor({12}));
    EXPECT_EQ(embedding.forward(tokens, output, cpu_context()).code(), base::InvalidArgument);

    embedding.set_param(0, weight);
    auto wrong_size_output = cpu_tensor({5});
    EXPECT_EQ(embedding.forward(tokens, wrong_size_output, cpu_context()).code(),
              base::InvalidArgument);

    op::OpContext unknown_context;
    EXPECT_EQ(embedding.forward(tokens, output, unknown_context).code(), base::InvalidArgument);

    auto gpu_context = cpu_context();
    gpu_context._device_type = base::DeviceType::GPU;
    EXPECT_EQ(embedding.forward(tokens, output, gpu_context).code(), base::InvalidArgument);
}

class EmbeddingCudaTest : public ::testing::Test {
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

TEST_F(EmbeddingCudaTest, bf16_weights_produce_fp32_embeddings_on_non_default_stream) {
    check_bf16_embedding(base::DeviceType::GPU, stream);
}

TEST_F(EmbeddingCudaTest, forward_handles_unaligned_rows_and_scalar_tail) {
    ASSERT_EQ(kernel::get_embedding_kernel(base::DeviceType::GPU), kernel::emb_kernel_cu)
        << "the GPU embedding kernel selector must be implemented before forward can run";

    constexpr int32_t vocab_size = 6;
    constexpr int32_t embedding_dim = 5;
    const std::vector<int32_t> token_values = {5, 1, 0};
    const auto weight_values = make_weight_values(vocab_size, embedding_dim);
    std::vector<float> output_values(token_values.size() * embedding_dim);

    auto allocator = base::GPUAllocatorFactory::get_instance();
    tensor::Tensor tokens(base::DataType::Int32,
                          {static_cast<int32_t>(token_values.size())}, allocator);
    tensor::Tensor weight(base::DataType::Fp32, {vocab_size, embedding_dim}, allocator);
    tensor::Tensor output(
        base::DataType::Fp32,
        {static_cast<int32_t>(token_values.size()), embedding_dim}, allocator);

    ASSERT_EQ(cudaMemcpyAsync(tokens.ptr<int32_t>(), token_values.data(), tokens.byte_size(),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(weight.ptr<float>(), weight_values.data(), weight.byte_size(),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);

    op::EmbeddingOp embedding;
    embedding.reset_param_size(1);
    embedding.set_param(0, weight);
    op::OpContext context;
    context._device_type = base::DeviceType::GPU;
    context._stream = stream;

    const auto status = embedding.forward(tokens, output, context);

    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(output_values.data(), output.ptr<float>(), output.byte_size(),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expect_selected_rows(token_values, weight_values, embedding_dim, output_values.data());
}

} // namespace
