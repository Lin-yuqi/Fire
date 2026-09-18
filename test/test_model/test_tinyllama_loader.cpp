#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "Fire/base/alloc.h"
#include "Fire/model/tinyllama.h"
#include "Fire/model/tinyllama_loader.h"
#include "Fire/op/rmsnorm.h"
#include "Fire/tensor/tensor.h"

TEST(TinyllamaTest, CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache) {
    if (std::getenv("FIRE_RUN_TINYLLAMA_FORWARD_TEST") == nullptr) {
        GTEST_SKIP() << "set FIRE_RUN_TINYLLAMA_FORWARD_TEST=1 to run the 4.1GB model";
    }
    if (!std::filesystem::exists(FIRE_TINYLLAMA_PATH)) {
        GTEST_SKIP() << "TinyLlama .fire file is unavailable: " << FIRE_TINYLLAMA_PATH;
    }

    model::TinyllamaLoader loader;
    auto status = loader.open(FIRE_TINYLLAMA_PATH);
    ASSERT_TRUE(status.ok()) << status.message();

    model::TinyLlamaWeights weights;
    status = loader.load_weights(weights);
    ASSERT_TRUE(status.ok()) << status.message();

    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    context._allocator = base::CPUAllocatorFactory::get_instance();

    std::unique_ptr<model::TinyLlamaModel> tinyllama;
    status = model::TinyLlamaModel::create(weights, context, tinyllama);
    ASSERT_TRUE(status.ok()) << status.message();

    status = tinyllama->prepare(2, context);
    ASSERT_TRUE(status.ok()) << status.message();

    tensor::Tensor logits(base::DataType::Fp32,
                          {model::TinyLlamaProfile::model.vocab_size}, context._allocator);

    status = tinyllama->forward(1, 0, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();
    std::vector<float> first_logits(logits.ptr<float>(), logits.ptr<float>() + logits.size());

    status = tinyllama->forward(1, 1, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();

    bool logits_changed = false;
    float max_abs_logit = 0.0f;
    size_t top_token = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(first_logits[i])) << "first forward logit index = " << i;
        ASSERT_TRUE(std::isfinite(logits.ptr<float>()[i])) << "second forward logit index = " << i;
        logits_changed = logits_changed || first_logits[i] != logits.ptr<float>()[i];
        if (std::abs(logits.ptr<float>()[i]) > max_abs_logit) {
            max_abs_logit = std::abs(logits.ptr<float>()[i]);
            top_token = i;
        }
    }
    EXPECT_TRUE(logits_changed);
    EXPECT_GT(max_abs_logit, 0.0f);
    std::cout << "second forward max |logit| = " << max_abs_logit
              << ", token index = " << top_token << '\n';

    EXPECT_EQ(tinyllama->forward(1, 1, logits, context).code(), base::InvalidArgument);
}

TEST(TinyllamaTest, GpuForwardTwoTokensOnNonDefaultStream) {
    if (std::getenv("FIRE_RUN_TINYLLAMA_GPU_FORWARD_TEST") == nullptr) {
        GTEST_SKIP() << "set FIRE_RUN_TINYLLAMA_GPU_FORWARD_TEST=1 to run the 4.1GB model";
    }
    if (!std::filesystem::exists(FIRE_TINYLLAMA_PATH)) {
        GTEST_SKIP() << "TinyLlama .fire file is unavailable: " << FIRE_TINYLLAMA_PATH;
    }

    int32_t device_count = 0;
    const auto device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice || device_status == cudaErrorInsufficientDriver ||
        (device_status == cudaSuccess && device_count == 0)) {
        GTEST_SKIP() << "CUDA unavailable";
    }
    ASSERT_EQ(device_status, cudaSuccess) << cudaGetErrorString(device_status);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    constexpr size_t required_free_bytes = 5ULL * 1024 * 1024 * 1024;
    if (free_bytes < required_free_bytes) {
        GTEST_SKIP() << "TinyLlama GPU forward requires at least 5 GiB free VRAM";
    }

    model::TinyllamaLoader loader;
    auto status = loader.open(FIRE_TINYLLAMA_PATH);
    ASSERT_TRUE(status.ok()) << status.message();

    model::TinyLlamaWeights weights;
    status = loader.load_weights(weights);
    ASSERT_TRUE(status.ok()) << status.message();

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    op::OpContext context;
    context._device_type = base::DeviceType::GPU;
    context._allocator = base::GPUAllocatorFactory::get_instance();
    context._stream = stream;

    std::unique_ptr<model::TinyLlamaModel> tinyllama;
    status = model::TinyLlamaModel::create(weights, context, tinyllama);
    ASSERT_TRUE(status.ok()) << status.message();

    status = tinyllama->prepare(2, context);
    ASSERT_TRUE(status.ok()) << status.message();

    tensor::Tensor logits(base::DataType::Fp32,
                          {model::TinyLlamaProfile::model.vocab_size}, context._allocator);
    std::vector<float> first_logits(logits.size());
    std::vector<float> second_logits(logits.size());

    status = tinyllama->forward(1, 0, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(cudaMemcpyAsync(first_logits.data(), logits.ptr<float>(), logits.byte_size(),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    status = tinyllama->forward(1, 1, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_EQ(cudaMemcpyAsync(second_logits.data(), logits.ptr<float>(), logits.byte_size(),
                              cudaMemcpyDeviceToHost, stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    bool logits_changed = false;
    float max_abs_logit = 0.0f;
    size_t top_token = 0;
    for (size_t i = 0; i < second_logits.size(); ++i) {
        ASSERT_TRUE(std::isfinite(first_logits[i])) << "first forward logit index = " << i;
        ASSERT_TRUE(std::isfinite(second_logits[i])) << "second forward logit index = " << i;
        logits_changed = logits_changed || first_logits[i] != second_logits[i];
        if (std::abs(second_logits[i]) > max_abs_logit) {
            max_abs_logit = std::abs(second_logits[i]);
            top_token = i;
        }
    }
    EXPECT_TRUE(logits_changed);
    EXPECT_GT(max_abs_logit, 0.0f);
    std::cout << "GPU second forward max |logit| = " << max_abs_logit
              << ", token index = " << top_token << '\n';
    EXPECT_EQ(tinyllama->forward(1, 1, logits, context).code(), base::InvalidArgument);

    op::OpContext cpu_context;
    cpu_context._device_type = base::DeviceType::CPU;
    cpu_context._allocator = base::CPUAllocatorFactory::get_instance();
    std::unique_ptr<model::TinyLlamaModel> cpu_tinyllama;
    status = model::TinyLlamaModel::create(weights, cpu_context, cpu_tinyllama);
    ASSERT_TRUE(status.ok()) << status.message();
    status = cpu_tinyllama->prepare(2, cpu_context);
    ASSERT_TRUE(status.ok()) << status.message();

    tensor::Tensor cpu_logits(base::DataType::Fp32,
                              {model::TinyLlamaProfile::model.vocab_size}, cpu_context._allocator);
    status = cpu_tinyllama->forward(1, 0, cpu_logits, cpu_context);
    ASSERT_TRUE(status.ok()) << status.message();
    float first_max_abs_diff = 0.0f;
    for (size_t i = 0; i < first_logits.size(); ++i) {
        first_max_abs_diff =
            std::max(first_max_abs_diff, std::abs(first_logits[i] - cpu_logits.ptr<float>()[i]));
    }

    status = cpu_tinyllama->forward(1, 1, cpu_logits, cpu_context);
    ASSERT_TRUE(status.ok()) << status.message();
    float second_max_abs_diff = 0.0f;
    for (size_t i = 0; i < second_logits.size(); ++i) {
        second_max_abs_diff =
            std::max(second_max_abs_diff, std::abs(second_logits[i] - cpu_logits.ptr<float>()[i]));
    }
    EXPECT_LT(first_max_abs_diff, 1e-2f);
    EXPECT_LT(second_max_abs_diff, 1e-2f);
    std::cout << "CPU/GPU max |logit diff|: pos0 = " << first_max_abs_diff
              << ", pos1 = " << second_max_abs_diff << '\n';

    tinyllama.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

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
