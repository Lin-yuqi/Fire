#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "Fire/tensor/tensor.h"
#include <gtest/gtest.h>
#include <Fire/op/add.h>
#include "../utils.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace {

    
TEST(op_test, add) {
    int count = 0;
    const auto error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(error);
    }
    ASSERT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    auto alloc_cu = base::GPUAllocatorFactory::get_instance();
    int32_t sz = 32 * 129;
    tensor::Tensor t1(base::DataType::Fp32, {sz}, alloc_cu);
    tensor::Tensor t2(base::DataType::Fp32, {sz}, alloc_cu);
    tensor::Tensor t3(base::DataType::Fp32, {sz}, alloc_cu);

    set_value_cu(t1.ptr<float>(), sz, 2.f);
    set_value_cu(t2.ptr<float>(), sz, 3.f);

    op::OpContext opctext;
    opctext._device_type = base::DeviceType::GPU;

    op::VecAddOp add;
    ASSERT_TRUE(add.forward(t1, t2, t3, opctext).ok());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> output(sz);
    ASSERT_EQ(
        cudaMemcpy(output.data(), t3.ptr<float>(), sz * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (int i = 0; i < sz; i++) {
        ASSERT_EQ(output[i], 5.f);
    }
}

tensor::Tensor cpu_tensor(std::vector<int32_t> dims, base::DataType dtype = base::DataType::Fp32) {
    return tensor::Tensor(dtype, dims, base::CPUAllocatorFactory::get_instance());
}

class CheckedOperator : public op::Operator {
  public:
    using Operator::_check_tensor;
    using Operator::_check_tensor_with_dim;
    using Operator::Operator;
};

TEST(op_test, metadata_and_validation) {
    CheckedOperator operation(op::OpType::Add, "sum");
    EXPECT_EQ(operation.type(), op::OpType::Add);
    EXPECT_EQ(operation.name(), "sum");
    operation.set_name("residual");
    EXPECT_EQ(operation.name(), "residual");
    auto tensor = cpu_tensor({2, 3});
    EXPECT_TRUE(operation._check_tensor(tensor, base::DeviceType::CPU, base::DataType::Fp32).ok());
    EXPECT_EQ(operation._check_tensor({}, base::DeviceType::CPU, base::DataType::Fp32).code(),
              base::InvalidArgument);
    EXPECT_EQ(operation._check_tensor(tensor, base::DeviceType::GPU, base::DataType::Fp32).code(),
              base::InvalidArgument);
    EXPECT_EQ(operation._check_tensor(tensor, base::DeviceType::CPU, base::DataType::int32).code(),
              base::InvalidArgument);
    EXPECT_TRUE(
        operation
            ._check_tensor_with_dim(tensor, base::DeviceType::CPU, base::DataType::Fp32, {2, 3})
            .ok());
    EXPECT_EQ(
        operation._check_tensor_with_dim(tensor, base::DeviceType::CPU, base::DataType::Fp32, {6})
            .code(),
        base::InvalidArgument);
    EXPECT_EQ(
        operation
            ._check_tensor_with_dim(tensor, base::DeviceType::CPU, base::DataType::Fp32, {3, 2})
            .code(),
        base::InvalidArgument);
}

TEST(op_test, parameter_storage_and_resize) {
    op::ParamOperator operation(op::OpType::Linear);
    EXPECT_EQ(operation.param_size(), 0u);
    operation.reset_param_size(2);
    op::Parameter parameter;
    EXPECT_FALSE(parameter.is_quantized());
    parameter._data = cpu_tensor({1});
    parameter._data.ptr<float>()[0] = 7.f;
    parameter._quant_config._quant_type = op::QuantType::Int8PerTensor;
    operation.set_param(0, parameter);
    const auto& read_only = operation;
    EXPECT_TRUE(read_only.get_param(0).is_quantized());
    EXPECT_FLOAT_EQ(read_only.get_param(0)._data.ptr<float>()[0], 7.f);
    operation.get_param(0)._quant_config._group_size = 32;
    operation.reset_param_size(3);
    EXPECT_EQ(read_only.get_param(0)._quant_config._group_size, 32);
    EXPECT_FALSE(read_only.get_param(2).is_quantized());
    operation.reset_param_size(1);
    EXPECT_EQ(operation.param_size(), 1u);
    EXPECT_FLOAT_EQ(read_only.get_param(0)._data.ptr<float>()[0], 7.f);
    operation.reset_param_size(0);
    EXPECT_EQ(operation.param_size(), 0u);
}

TEST(op_death_test, parameter_index_out_of_range) {
    op::ParamOperator operation(op::OpType::Linear);
    operation.reset_param_size(1);
    const auto& read_only = operation;
    EXPECT_DEATH(operation.get_param(1), "Check failed");
    EXPECT_DEATH(read_only.get_param(1), "Check failed");
    EXPECT_DEATH(operation.set_param(1, op::Parameter{}), "Check failed");
}

TEST(add_test, cpu_forward_preserves_inputs) {
    for (int size : {1, 511, 512, 513, 4128}) {
        SCOPED_TRACE(size);
        auto a = cpu_tensor({size});
        auto b = cpu_tensor({size});
        auto out = cpu_tensor({size});
        for (int i = 0; i < size; ++i) {
            a.ptr<float>()[i] = float(i % 17) - 8.f;
            b.ptr<float>()[i] = float(i % 7) * 0.25f;
            out.ptr<float>()[i] = -999.f;
        }
        op::OpContext context;
        context._device_type = base::DeviceType::CPU;
        op::VecAddOp add;
        const auto status = add.forward(a, b, out, context);
        ASSERT_TRUE(status.ok()) << status.message();
        for (int i = 0; i < size; ++i) {
            EXPECT_FLOAT_EQ(out.ptr<float>()[i], float(i % 17) - 8.f + float(i % 7) * 0.25f) << i;
            EXPECT_FLOAT_EQ(a.ptr<float>()[i], float(i % 17) - 8.f);
            EXPECT_FLOAT_EQ(b.ptr<float>()[i], float(i % 7) * 0.25f);
        }
    }
}

TEST(add_test, multidimensional_shape) {
    auto a = cpu_tensor({2, 3});
    auto b = cpu_tensor({2, 3});
    auto out = cpu_tensor({2, 3});
    for (int i = 0; i < 6; ++i) {
        a.ptr<float>()[i] = float(i);
        b.ptr<float>()[i] = -2.f;
    }
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    op::VecAddOp add;
    ASSERT_TRUE(add.forward(a, b, out, context).ok());
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(out.ptr<float>()[i], float(i) - 2.f);
}

TEST(add_test, rejects_invalid_tensor_in_each_position) {
    op::VecAddOp add;
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    for (int position = 0; position < 3; ++position) {
        for (int kind = 0; kind < 5; ++kind) {
            SCOPED_TRACE(::testing::Message() << "position=" << position << " kind=" << kind);
            std::vector<tensor::Tensor> tensors = {cpu_tensor({2, 3}), cpu_tensor({2, 3}),
                                                   cpu_tensor({2, 3})};
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
            case 4:
                invalid.set_device_type(base::DeviceType::GPU);
                break;
            }
            const auto status = add.forward(tensors[0], tensors[1], tensors[2], context);
            EXPECT_EQ(status.code(), base::InvalidArgument);
            EXPECT_FALSE(status.message().empty());
        }
    }
}

TEST(add_test, rejects_unset_or_mismatched_context) {
    auto a = cpu_tensor({1});
    auto b = cpu_tensor({1});
    auto out = cpu_tensor({1});
    op::VecAddOp add;
    op::OpContext context;
    EXPECT_EQ(add.forward(a, b, out, context).code(), base::InvalidArgument);
    context._device_type = base::DeviceType::GPU;
    EXPECT_EQ(add.forward(a, b, out, context).code(), base::InvalidArgument);
}

class AddCudaTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int count = 0;
        const auto error = cudaGetDeviceCount(&count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
            (error == cudaSuccess && count == 0)) {
            GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(error);
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

TEST_F(AddCudaTest, forward_on_non_default_stream_at_block_boundaries) {
    op::VecAddOp add;
    op::OpContext context;
    context._device_type = base::DeviceType::GPU;
    context._stream = stream;
    auto allocator = base::GPUAllocatorFactory::get_instance();
    for (int size : {1, 511, 512, 513, 4128}) {
        SCOPED_TRACE(size);
        std::vector<float> a(size), b(size), result(size);
        for (int i = 0; i < size; ++i) {
            a[i] = float(i % 17) - 8.f;
            b[i] = float(i % 7) * 0.25f;
        }
        tensor::Tensor input1(base::DataType::Fp32, {size}, allocator);
        tensor::Tensor input2(base::DataType::Fp32, {size}, allocator);
        tensor::Tensor output(base::DataType::Fp32, {size}, allocator);
        const auto bytes = size * sizeof(float);
        ASSERT_EQ(
            cudaMemcpyAsync(input1.ptr<float>(), a.data(), bytes, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpyAsync(input2.ptr<float>(), b.data(), bytes, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(output.ptr<float>(), 0xff, bytes, stream), cudaSuccess);
        ASSERT_TRUE(add.forward(input1, input2, output, context).ok());
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(result.data(), output.ptr<float>(), bytes, cudaMemcpyDeviceToHost,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        for (int i = 0; i < size; ++i)
            EXPECT_FLOAT_EQ(result[i], a[i] + b[i]) << i;
    }
}

} // namespace
