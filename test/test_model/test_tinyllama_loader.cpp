#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/model/tinyllama_loader.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/tensor/tensor.h"

TEST(TinyllamaTest, LoadRmsNormWeight) {
    // 没有 CUDA 就跳过
    int device_count = 0;
    auto error = cudaGetDeviceCount(&device_count);

    if (error == cudaErrorNoDevice ||
        error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && device_count == 0)) {
        GTEST_SKIP() << "CUDA unavailable";
    }

    ASSERT_EQ(error, cudaSuccess)
        << cudaGetErrorString(error);

    // 1. 打开真实 TinyLlama .fire
    model::TinyllamaLoader loader;

    auto status = loader.open(FIRE_TINYLLAMA_PATH);
    ASSERT_TRUE(status.ok()) << status.message();

    // 2. 从 mmap 中拿到 RMSNorm weight
    tensor::Tensor weight;

    status = loader.loader_tensor(
        "layers.0.attention_norm.weight",
        weight
    );

    ASSERT_TRUE(status.ok()) << status.message();

    ASSERT_EQ(weight.data_type(), base::DataType::Fp32);
    ASSERT_EQ(weight.device_type(), base::DeviceType::CPU);

    ASSERT_EQ(weight.dims().size(), 1);
    ASSERT_EQ(weight.dims()[0], 2048);

    ASSERT_FALSE(weight.is_empty());

    // 先保存 CPU 权重，后面用于验证 GPU 结果
    std::vector<float> weight_cpu(
        weight.ptr<float>(),
        weight.ptr<float>() + weight.size()
    );

    // 3. 权重搬到 GPU
    weight.to_cuda();

    ASSERT_EQ(weight.device_type(), base::DeviceType::GPU);

    // 4. 构造一个全 1 输入
    constexpr int32_t size = 2048;
    constexpr float eps = 1e-5f;

    std::vector<float> input_cpu(size, 1.0f);
    std::vector<float> output_cpu(size);

    auto allocator = base::GPUAllocatorFactory::get_instance();

    tensor::Tensor input(
        base::DataType::Fp32,
        {size},
        allocator
    );

    tensor::Tensor output(
        base::DataType::Fp32,
        {size},
        allocator
    );

    ASSERT_EQ(
        cudaMemcpy(
            input.ptr<float>(),
            input_cpu.data(),
            input.byte_size(),
            cudaMemcpyHostToDevice
        ),
        cudaSuccess
    );

    // 5. 用真实 TinyLlama weight 跑 RMSNorm
    op::RmsNormOp rmsnorm(eps);

    rmsnorm.reset_param_size(1);
    rmsnorm.get_param(0)._data = weight;

    op::OpContext context;
    context._device_type = base::DeviceType::GPU;

    status = rmsnorm.forward(input, output, context);

    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // 6. 拷回 CPU
    ASSERT_EQ(
        cudaMemcpy(
            output_cpu.data(),
            output.ptr<float>(),
            output.byte_size(),
            cudaMemcpyDeviceToHost
        ),
        cudaSuccess
    );

    // input 全是 1:
    //
    // rms = sqrt(mean(1^2) + eps)
    //     = sqrt(1 + eps)
    //
    // output[i] = weight[i] / sqrt(1 + eps)
    const float scale = 1.0f / std::sqrt(1.0f + eps);

    for (int32_t i = 0; i < size; ++i) {
        EXPECT_NEAR(
            output_cpu[i],
            weight_cpu[i] * scale,
            1e-5f
        ) << "index = " << i;
    }
}