#include "Fire/base/alloc.h"
#include "Fire/model/qwen.h"
#include "Fire/model/qwen_loader.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

model::Qwen3Profile SmallQwenProfile() {
    return {{"test/Qwen3", 128, 128}, 128, 128, 1, 1, 1, 128, 1e-6f, 10000.0f, false};
}

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (size_t index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<uint8_t>(value >> (8 * index));
    }
}

void WriteU64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        bytes[offset + index] = static_cast<uint8_t>(value >> (8 * index));
    }
}

std::vector<uint8_t> SmallQwenFixture(bool quantized, bool wrong_group = false) {
    struct Entry {
        std::string name;
        uint8_t dtype;
        std::vector<int32_t> dims;
        uint32_t group_size = 0;
    };
    std::vector<Entry> entries;
    const auto add_plain = [&](std::string name, std::vector<int32_t> dims) {
        entries.push_back({std::move(name), 1, std::move(dims)});
    };
    const auto add_linear = [&](std::string name, int32_t rows, int32_t columns) {
        if (!quantized) {
            add_plain(std::move(name), {rows, columns});
            return;
        }
        const std::string prefix = name.substr(0, name.size() - 7); // remove .weight
        entries.push_back({prefix + ".qweight", 2, {rows, columns / 2},
                           static_cast<uint32_t>(wrong_group ? 64 : 128)});
        entries.push_back({prefix + ".scales", 1, {rows, columns / 128}});
        entries.push_back({prefix + ".zero_points", 2, {rows, columns / 128}});
    };

    add_plain("tok_embeddings.weight", {128, 128});
    add_plain("layers.0.attention_norm.weight", {128});
    add_linear("layers.0.attention.wq.weight", 128, 128);
    add_linear("layers.0.attention.wk.weight", 128, 128);
    add_linear("layers.0.attention.wv.weight", 128, 128);
    add_linear("layers.0.attention.wo.weight", 128, 128);
    add_plain("layers.0.attention.q_norm.weight", {128});
    add_plain("layers.0.attention.k_norm.weight", {128});
    add_plain("layers.0.ffn_norm.weight", {128});
    add_linear("layers.0.feed_forward.w1.weight", 128, 128);
    add_linear("layers.0.feed_forward.w2.weight", 128, 128);
    add_linear("layers.0.feed_forward.w3.weight", 128, 128);
    add_plain("norm.weight", {128});
    add_linear("output.weight", 128, 128);

    std::vector<uint8_t> bytes(32 + entries.size() * 96, 0);
    std::memcpy(bytes.data(), "FIRECKPT", 8);
    WriteU32(bytes, 8, quantized ? 2 : 1);
    WriteU32(bytes, 12, static_cast<uint32_t>(entries.size()));
    WriteU64(bytes, 16, 32);
    WriteU64(bytes, 24, bytes.size());
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        bytes.resize((bytes.size() + 3) & ~size_t{3}, 0);
        const size_t offset = 32 + index * 96;
        std::memcpy(bytes.data() + offset, entry.name.data(), entry.name.size());
        const size_t elements = static_cast<size_t>(entry.dims[0]) *
                                (entry.dims.size() == 2 ? entry.dims[1] : 1);
        const size_t size = elements * (entry.dtype == 1 ? 4 : 1);
        WriteU64(bytes, offset + 64, bytes.size());
        WriteU64(bytes, offset + 72, size);
        WriteU32(bytes, offset + 80, entry.dims[0]);
        WriteU32(bytes, offset + 84, entry.dims.size() == 2 ? entry.dims[1] : 0);
        bytes[offset + 88] = entry.dtype;
        bytes[offset + 89] = static_cast<uint8_t>(entry.dims.size());
        if (entry.group_size != 0) {
            bytes[offset + 90] = 1;
            WriteU32(bytes, offset + 91, entry.group_size);
        }
        bytes.resize(bytes.size() + size, 0);
    }
    return bytes;
}

class TemporaryQwenFile {
  public:
    explicit TemporaryQwenFile(const std::vector<uint8_t>& bytes) {
        std::array<char, 32> pattern{};
        std::memcpy(pattern.data(), "/tmp/qwen_loader_XXXXXX", 24);
        const int descriptor = ::mkstemp(pattern.data());
        if (descriptor < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        ::close(descriptor);
        _path = pattern.data();
        std::ofstream output(_path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            throw std::runtime_error("failed to write Qwen3 fixture");
        }
    }

    ~TemporaryQwenFile() { std::remove(_path.c_str()); }
    const std::string& path() const { return _path; }

  private:
    std::string _path;
};

} // namespace

TEST(Qwen3LoaderTest, LoadsFp32AndInt4Parameters) {
    const auto profile = SmallQwenProfile();
    for (const bool quantized : {false, true}) {
        TemporaryQwenFile file(SmallQwenFixture(quantized));
        model::Qwen3Loader loader(profile);
        auto status = loader.open(file.path());
        ASSERT_TRUE(status.ok()) << status.message();
        model::Qwen3Weights weights;
        status = loader.load_weights(weights);
        ASSERT_TRUE(status.ok()) << status.message();

        const auto& weight = weights.layers.front().wq;
        EXPECT_EQ(weight.is_quantized(), quantized);
        EXPECT_EQ(weight._data.data_type(),
                  quantized ? base::DataType::UInt8 : base::DataType::Fp32);
        EXPECT_EQ(weight._data.dims(), (std::vector<int32_t>{128, quantized ? 64 : 128}));
        EXPECT_EQ(weight._scales.is_empty(), !quantized);
        EXPECT_EQ(weight._zero_points.is_empty(), !quantized);
        EXPECT_EQ(weights.output.is_quantized(), quantized);
        if (quantized) {
            EXPECT_EQ(weight._scales.dims(), (std::vector<int32_t>{128, 1}));
            EXPECT_EQ(weight._zero_points.dims(), (std::vector<int32_t>{128, 1}));
            EXPECT_EQ(weight._quant_config._quant_type, op::QuantType::Int4GroupWise);
            EXPECT_EQ(weight._quant_config._group_size, 128);
            EXPECT_FALSE(weight._quant_config._symmetric);
        }
    }
}

TEST(Qwen3LoaderTest, RejectsUnsupportedInt4GroupSize) {
    TemporaryQwenFile file(SmallQwenFixture(true, true));
    model::Qwen3Loader loader(SmallQwenProfile());
    auto status = loader.open(file.path());
    ASSERT_TRUE(status.ok()) << status.message();

    model::Qwen3Weights weights;
    status = loader.load_weights(weights);
    EXPECT_EQ(status.code(), base::StatusCode::ModelParseError);
    EXPECT_TRUE(weights.layers.empty());
}
TEST(Qwen3ProfileTest, PresetsKeepArchitectureDifferencesInData) {
    const auto& small = model::qwen3_profiles::Qwen3_0_6B;
    const auto& large = model::qwen3_profiles::Qwen3_8B;

    ASSERT_TRUE(small.is_valid());
    ASSERT_TRUE(large.is_valid());
    EXPECT_EQ(small.q_dim(), 2048);
    EXPECT_NE(small.q_dim(), small.hidden_size);
    EXPECT_EQ(large.q_dim(), large.hidden_size);
    EXPECT_EQ(small.kv_dim(), 1024);
    EXPECT_EQ(large.kv_dim(), 1024);
    EXPECT_TRUE(small.tie_word_embeddings);
    EXPECT_FALSE(large.tie_word_embeddings);
    EXPECT_EQ(small.tensor_count(), 311U);
    EXPECT_EQ(large.tensor_count(), 399U);
}

TEST(Qwen3ProfileTest, RejectsStructurallyInvalidProfiles) {
    auto profile = model::qwen3_profiles::Qwen3_0_6B;
    profile.num_attention_heads = 15;
    EXPECT_FALSE(profile.is_valid());

    profile = model::qwen3_profiles::Qwen3_0_6B;
    profile.head_dim = 127;
    EXPECT_FALSE(profile.is_valid());
}

TEST(Qwen3BlockTest, AllNormsUseTheProfileEpsilon) {
    constexpr float epsilon = model::qwen3_profiles::Qwen3_0_6B.rms_norm_eps;
    auto allocator = base::CPUAllocatorFactory::get_instance();
    tensor::Tensor weight(base::DataType::Fp32, {2}, allocator);
    weight.ptr<float>()[0] = 1.0f;
    weight.ptr<float>()[1] = 1.0f;
    tensor::Tensor input(base::DataType::Fp32, {2}, allocator);
    input.ptr<float>()[0] = 0.001f;
    input.ptr<float>()[1] = 0.002f;
    tensor::Tensor output(base::DataType::Fp32, {2}, allocator);
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;

    model::Qwen3Block block(epsilon);
    for (auto* norm :
         {&block.attention_norm, &block.q_norm, &block.k_norm, &block.ffn_norm}) {
        norm->reset_param_size(1);
        norm->get_param(0)._data = weight;
        const auto status = norm->forward(input, output, context);
        ASSERT_TRUE(status.ok()) << status.message();
        const float inverse_rms = 1.0f / std::sqrt(2.5e-6f + epsilon);
        EXPECT_NEAR(output.ptr<float>()[0], 0.001f * inverse_rms, 1e-6f);
        EXPECT_NEAR(output.ptr<float>()[1], 0.002f * inverse_rms, 1e-6f);
    }
}

TEST(Qwen3ModelTest, CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache) {
    if (std::getenv("FIRE_RUN_QWEN3_FORWARD_TEST") == nullptr) {
        GTEST_SKIP() << "set FIRE_RUN_QWEN3_FORWARD_TEST=1 to run the 2.8 GiB model";
    }
    if (!std::filesystem::exists(FIRE_QWEN3_0_6B_PATH)) {
        GTEST_SKIP() << "Qwen3-0.6B .fire file is unavailable: " << FIRE_QWEN3_0_6B_PATH;
    }

    const auto& profile = model::qwen3_profiles::Qwen3_0_6B;
    model::Qwen3Loader loader(profile);
    auto status = loader.open(FIRE_QWEN3_0_6B_PATH);
    ASSERT_TRUE(status.ok()) << status.message();

    model::Qwen3Weights weights;
    status = loader.load_weights(weights);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(weights.layers.front().wq.is_quantized());
    EXPECT_EQ(weights.layers.front().wq._data.data_type(), base::DataType::Fp32);
    EXPECT_TRUE(weights.layers.front().wq._scales.is_empty());
    EXPECT_TRUE(weights.layers.front().wq._zero_points.is_empty());
    EXPECT_FALSE(weights.output.is_quantized());

    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    context._allocator = base::CPUAllocatorFactory::get_instance();

    std::unique_ptr<model::Qwen3Model> qwen3;
    status = model::Qwen3Model::create(weights, context, qwen3);
    ASSERT_TRUE(status.ok()) << status.message();

    status = qwen3->prepare(2, context);
    ASSERT_TRUE(status.ok()) << status.message();

    tensor::Tensor logits(base::DataType::Fp32, {profile.model.vocab_size}, context._allocator);
    status = qwen3->forward(1, 0, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();
    std::vector<float> first_logits(logits.ptr<float>(), logits.ptr<float>() + logits.size());

    status = qwen3->forward(1, 1, logits, context);
    ASSERT_TRUE(status.ok()) << status.message();

    bool logits_changed = false;
    float max_abs_logit = 0.0f;
    for (size_t index = 0; index < logits.size(); ++index) {
        ASSERT_TRUE(std::isfinite(first_logits[index])) << "first logit index = " << index;
        ASSERT_TRUE(std::isfinite(logits.ptr<float>()[index]))
            << "second logit index = " << index;
        logits_changed = logits_changed || first_logits[index] != logits.ptr<float>()[index];
        max_abs_logit = std::max(max_abs_logit, std::abs(logits.ptr<float>()[index]));
    }
    EXPECT_TRUE(logits_changed);
    EXPECT_GT(max_abs_logit, 0.0f);
    EXPECT_EQ(qwen3->forward(1, 1, logits, context).code(), base::InvalidArgument);
}

TEST(Qwen3ModelTest, GpuForwardTwoTokensMatchesCpuOnNonDefaultStream) {
    if (std::getenv("FIRE_RUN_QWEN3_GPU_FORWARD_TEST") == nullptr) {
        GTEST_SKIP() << "set FIRE_RUN_QWEN3_GPU_FORWARD_TEST=1 to run the 2.8 GiB model";
    }
    if (!std::filesystem::exists(FIRE_QWEN3_0_6B_PATH)) {
        GTEST_SKIP() << "Qwen3-0.6B .fire file is unavailable: " << FIRE_QWEN3_0_6B_PATH;
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
    constexpr size_t required_free_bytes = 4ULL * 1024 * 1024 * 1024;
    if (free_bytes < required_free_bytes) {
        GTEST_SKIP() << "Qwen3-0.6B GPU forward requires at least 4 GiB free VRAM";
    }

    const auto& profile = model::qwen3_profiles::Qwen3_0_6B;
    model::Qwen3Loader loader(profile);
    auto status = loader.open(FIRE_QWEN3_0_6B_PATH);
    ASSERT_TRUE(status.ok()) << status.message();

    model::Qwen3Weights weights;
    status = loader.load_weights(weights);
    ASSERT_TRUE(status.ok()) << status.message();

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    op::OpContext gpu_context;
    gpu_context._device_type = base::DeviceType::GPU;
    gpu_context._allocator = base::GPUAllocatorFactory::get_instance();
    gpu_context._stream = stream;

    std::unique_ptr<model::Qwen3Model> gpu_model;
    status = model::Qwen3Model::create(weights, gpu_context, gpu_model);
    ASSERT_TRUE(status.ok()) << status.message();
    status = gpu_model->prepare(2, gpu_context);
    ASSERT_TRUE(status.ok()) << status.message();

    const std::vector<int32_t> token_ids{151643, 9707};
    const size_t vocab_size = static_cast<size_t>(profile.model.vocab_size);
    tensor::Tensor gpu_logits(base::DataType::Fp32, {profile.model.vocab_size},
                              gpu_context._allocator);
    std::vector<float> actual(token_ids.size() * vocab_size);
    for (size_t position = 0; position < token_ids.size(); ++position) {
        status = gpu_model->forward(token_ids[position], static_cast<int32_t>(position),
                                    gpu_logits, gpu_context);
        ASSERT_TRUE(status.ok()) << status.message();
        ASSERT_EQ(cudaMemcpyAsync(actual.data() + position * vocab_size,
                                  gpu_logits.ptr<float>(), gpu_logits.byte_size(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    op::OpContext cpu_context;
    cpu_context._device_type = base::DeviceType::CPU;
    cpu_context._allocator = base::CPUAllocatorFactory::get_instance();
    std::unique_ptr<model::Qwen3Model> cpu_model;
    status = model::Qwen3Model::create(weights, cpu_context, cpu_model);
    ASSERT_TRUE(status.ok()) << status.message();
    status = cpu_model->prepare(2, cpu_context);
    ASSERT_TRUE(status.ok()) << status.message();

    tensor::Tensor cpu_logits(base::DataType::Fp32, {profile.model.vocab_size},
                              cpu_context._allocator);
    float max_abs_error = 0.0f;
    for (size_t position = 0; position < token_ids.size(); ++position) {
        status = cpu_model->forward(token_ids[position], static_cast<int32_t>(position),
                                    cpu_logits, cpu_context);
        ASSERT_TRUE(status.ok()) << status.message();

        size_t cpu_argmax = 0;
        size_t gpu_argmax = 0;
        for (size_t index = 0; index < vocab_size; ++index) {
            const float expected = cpu_logits.ptr<float>()[index];
            const float observed = actual[position * vocab_size + index];
            ASSERT_TRUE(std::isfinite(observed))
                << "position = " << position << ", logit index = " << index;
            max_abs_error = std::max(max_abs_error, std::abs(expected - observed));
            if (expected > cpu_logits.ptr<float>()[cpu_argmax]) {
                cpu_argmax = index;
            }
            if (observed > actual[position * vocab_size + gpu_argmax]) {
                gpu_argmax = index;
            }
        }
        EXPECT_EQ(gpu_argmax, cpu_argmax) << "position = " << position;
    }
    EXPECT_LT(max_abs_error, 1e-2f);
    std::cout << "Qwen3-0.6B CPU/GPU max |logit diff| = " << max_abs_error << '\n';

    gpu_model.reset();
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
