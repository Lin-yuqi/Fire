#include "Fire/base/alloc.h"
#include "Fire/model/qwen.h"

#include <gtest/gtest.h>

#include <cmath>

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
