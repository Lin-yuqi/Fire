#include "Fire/base/alloc.h"
#include "Fire/model/qwen.h"
#include "Fire/model/qwen_loader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

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
