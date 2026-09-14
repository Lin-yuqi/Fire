#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
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

} // namespace
