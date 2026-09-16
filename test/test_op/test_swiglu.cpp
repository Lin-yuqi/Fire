#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/op/swiglu.h"
#include "Fire/tensor/tensor.h"
#include "cpu/swiglu_kernel.h"
#include "cuda/swiglu_kernel.cuh"
#include "kernels_interface.h"

namespace {

tensor::Tensor cpu_tensor(std::vector<int32_t> dims,
                          base::DataType dtype = base::DataType::Fp32) {
    return tensor::Tensor(dtype, std::move(dims),
                          base::CPUAllocatorFactory::get_instance());
}

op::OpContext cpu_context() {
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    return context;
}

float swiglu_reference(float gate, float up) {
    return gate / (1.0f + std::exp(-gate)) * up;
}

void fill_values(std::vector<float>& gate, std::vector<float>& up) {
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.375f;
        up[i] = static_cast<float>(static_cast<int32_t>(i % 11) - 5) * 0.25f;
    }
}

TEST(swiglu_kernel_interface_test, selects_kernel_for_each_supported_device) {
    EXPECT_EQ(kernel::get_swiglu_kernel(base::DeviceType::CPU), kernel::swiglu_kernel_cpu);
    EXPECT_EQ(kernel::get_swiglu_kernel(base::DeviceType::GPU), kernel::swiglu_kernel_cu);
}

TEST(swiglu_kernel_interface_death_test, rejects_unknown_device) {
    EXPECT_DEATH(
        {
            const auto selected = kernel::get_swiglu_kernel(base::DeviceType::Unknown);
            (void)selected;
        },
        "Unknown device type");
}

TEST(swiglu_test, cpu_forward_matches_reference_and_preserves_inputs) {
    const std::vector<float> gate_values = {-3.0f, -1.0f, 0.0f, 0.5f, 1.0f, 3.0f};
    const std::vector<float> up_values = {2.0f, -4.0f, 1.5f, 3.0f, -2.0f, 0.25f};

    auto gate = cpu_tensor({2, 3});
    auto up = cpu_tensor({2, 3});
    auto output = cpu_tensor({2, 3});
    std::copy(gate_values.begin(), gate_values.end(), gate.ptr<float>());
    std::copy(up_values.begin(), up_values.end(), up.ptr<float>());

    op::SwiGLUOp swiglu;
    auto context = cpu_context();
    const auto status = swiglu.forward(gate, up, output, context);

    ASSERT_TRUE(status.ok()) << status.message();
    for (size_t i = 0; i < gate_values.size(); ++i) {
        EXPECT_NEAR(output.ptr<float>()[i], swiglu_reference(gate_values[i], up_values[i]), 1e-6f)
            << "index = " << i;
        EXPECT_FLOAT_EQ(gate.ptr<float>()[i], gate_values[i]) << "index = " << i;
        EXPECT_FLOAT_EQ(up.ptr<float>()[i], up_values[i]) << "index = " << i;
    }
}

TEST(swiglu_test, supports_one_dimensional_input) {
    const std::vector<float> gate_values = {-2.0f, 0.0f, 2.0f};
    const std::vector<float> up_values = {0.5f, 4.0f, -1.5f};

    auto gate = cpu_tensor({3});
    auto up = cpu_tensor({3});
    auto output = cpu_tensor({3});
    std::copy(gate_values.begin(), gate_values.end(), gate.ptr<float>());
    std::copy(up_values.begin(), up_values.end(), up.ptr<float>());

    op::SwiGLUOp swiglu;
    auto context = cpu_context();
    const auto status = swiglu.forward(gate, up, output, context);

    ASSERT_TRUE(status.ok()) << status.message();
    for (size_t i = 0; i < gate_values.size(); ++i)
        EXPECT_NEAR(output.ptr<float>()[i], swiglu_reference(gate_values[i], up_values[i]), 1e-6f)
            << "index = " << i;
}

TEST(swiglu_test, rejects_invalid_tensor_in_each_position) {
    op::SwiGLUOp swiglu;
    auto context = cpu_context();

    for (int position = 0; position < 3; ++position) {
        for (int kind = 0; kind < 4; ++kind) {
            SCOPED_TRACE(::testing::Message() << "position=" << position << " kind=" << kind);
            std::vector<tensor::Tensor> tensors = {
                cpu_tensor({2, 3}), cpu_tensor({2, 3}), cpu_tensor({2, 3})};
            auto& invalid = tensors[position];
            switch (kind) {
            case 0:
                invalid = tensor::Tensor{};
                break;
            case 1:
                invalid = tensor::Tensor(base::DataType::Fp32, {2, 3});
                break;
            case 2:
                invalid = cpu_tensor({3, 2});
                break;
            case 3:
                invalid = cpu_tensor({2, 3}, base::DataType::int32);
                break;
            }

            const auto status = swiglu.forward(tensors[0], tensors[1], tensors[2], context);
            EXPECT_EQ(status.code(), base::InvalidArgument);
            EXPECT_FALSE(status.message().empty());
        }
    }
}

TEST(swiglu_test, rejects_unset_or_mismatched_context) {
    auto gate = cpu_tensor({4});
    auto up = cpu_tensor({4});
    auto output = cpu_tensor({4});
    op::SwiGLUOp swiglu;

    op::OpContext context;
    EXPECT_EQ(swiglu.forward(gate, up, output, context).code(), base::InvalidArgument);

    context._device_type = base::DeviceType::GPU;
    EXPECT_EQ(swiglu.forward(gate, up, output, context).code(), base::InvalidArgument);
}

class SwiGLUCudaTest : public ::testing::Test {
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

    void run_case(int32_t size) {
        std::vector<float> gate_values(size);
        std::vector<float> up_values(size);
        std::vector<float> output_values(size);
        std::vector<float> copied_gate(size);
        std::vector<float> copied_up(size);
        fill_values(gate_values, up_values);

        auto allocator = base::GPUAllocatorFactory::get_instance();
        tensor::Tensor gate(base::DataType::Fp32, {size}, allocator);
        tensor::Tensor up(base::DataType::Fp32, {size}, allocator);
        tensor::Tensor output(base::DataType::Fp32, {size}, allocator);
        ASSERT_EQ(cudaMemcpyAsync(gate.ptr<float>(), gate_values.data(), gate.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(up.ptr<float>(), up_values.data(), up.byte_size(),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);

        op::SwiGLUOp swiglu;
        op::OpContext context;
        context._device_type = base::DeviceType::GPU;
        context._stream = stream;
        const auto status = swiglu.forward(gate, up, output, context);

        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(output_values.data(), output.ptr<float>(), output.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(copied_gate.data(), gate.ptr<float>(), gate.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(copied_up.data(), up.ptr<float>(), up.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        for (int32_t i = 0; i < size; ++i) {
            EXPECT_NEAR(output_values[i], swiglu_reference(gate_values[i], up_values[i]), 1e-6f)
                << "index = " << i;
            EXPECT_FLOAT_EQ(copied_gate[i], gate_values[i]) << "index = " << i;
            EXPECT_FLOAT_EQ(copied_up[i], up_values[i]) << "index = " << i;
        }
    }

    cudaStream_t stream = nullptr;
};

TEST_F(SwiGLUCudaTest, forward_on_non_default_stream_at_block_boundaries) {
    for (const int32_t size : {1, 255, 256, 257, 1025}) {
        SCOPED_TRACE(size);
        run_case(size);
    }
}

} // namespace
