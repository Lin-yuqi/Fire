#include "Fire/model/model.h"
#include "Fire/model/tinyllama.h"
#include "Fire/model/tinyllama_loader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <type_traits>
#include <vector>

static_assert(std::is_abstract_v<model::Model>);
static_assert(!std::is_copy_constructible_v<model::TinyLlamaBlock>);
static_assert(std::is_move_constructible_v<model::TinyLlamaBlock>);
static_assert(std::is_move_assignable_v<model::TinyLlamaBlock>);

TEST(ModelStructureTest, LoadsTinyLlamaWeights) {
    if (!std::filesystem::exists(FIRE_TINYLLAMA_PATH)) {
        GTEST_SKIP() << "TinyLlama .fire file is unavailable: " << FIRE_TINYLLAMA_PATH;
    }

    model::TinyLlamaWeights weights;
    float final_norm_first_value = 0.0f;
    {
        model::TinyllamaLoader loader;
        auto status = loader.open(FIRE_TINYLLAMA_PATH);
        ASSERT_TRUE(status.ok()) << status.message();

        status = loader.load_weights(weights);
        ASSERT_TRUE(status.ok()) << status.message();

        ASSERT_EQ(weights.layers.size(),
                  static_cast<size_t>(model::TinyLlamaProfile::num_layers));
        EXPECT_EQ(weights.embedding.dims(),
                  (std::vector<int32_t>{model::TinyLlamaProfile::model.vocab_size,
                                        model::TinyLlamaProfile::hidden_size}));
        EXPECT_EQ(weights.layers.front().wq.dims(),
                  (std::vector<int32_t>{model::TinyLlamaProfile::hidden_size,
                                        model::TinyLlamaProfile::hidden_size}));
        EXPECT_EQ(weights.layers.back().w3.dims(),
                  (std::vector<int32_t>{model::TinyLlamaProfile::intermediate_size,
                                        model::TinyLlamaProfile::hidden_size}));
        EXPECT_EQ(weights.norm.dims(),
                  (std::vector<int32_t>{model::TinyLlamaProfile::hidden_size}));
        EXPECT_EQ(weights.output.dims(),
                  (std::vector<int32_t>{model::TinyLlamaProfile::model.vocab_size,
                                        model::TinyLlamaProfile::hidden_size}));

        EXPECT_EQ(weights.embedding.device_type(), base::DeviceType::CPU);
        EXPECT_FALSE(weights.embedding.is_empty());
        EXPECT_FALSE(weights.layers.back().w3.is_empty());
        EXPECT_FALSE(weights.output.is_empty());
        final_norm_first_value = weights.norm.ptr<float>()[0];
    }

    // Structured Tensor views own the shared mmap after Loader destruction.
    EXPECT_FLOAT_EQ(weights.norm.ptr<float>()[0], final_norm_first_value);
}

TEST(ModelStructureTest, BlockVectorGrowthPreservesBoundParameters) {
    auto allocator = base::CPUAllocatorFactory::get_instance();
    tensor::Tensor weight(base::DataType::Fp32, {2, 2}, allocator);
    weight.ptr<float>()[0] = 1.0f;
    weight.ptr<float>()[1] = 2.0f;
    weight.ptr<float>()[2] = 3.0f;
    weight.ptr<float>()[3] = 4.0f;

    std::vector<model::TinyLlamaBlock> blocks;
    blocks.emplace_back();
    blocks[0].wq.reset_param_size(1);
    blocks[0].wq.get_param(0)._data = weight;

    // Force relocation of an already-bound block, exercising the move chain
    // from TinyLlamaBlock through LinearOp and ParamOperator to Operator.
    blocks.reserve(blocks.capacity() + 1);

    tensor::Tensor input(base::DataType::Fp32, {2}, allocator);
    input.ptr<float>()[0] = 2.0f;
    input.ptr<float>()[1] = 3.0f;
    tensor::Tensor output(base::DataType::Fp32, {2}, allocator);
    op::OpContext context;
    context._device_type = base::DeviceType::CPU;

    const auto status = blocks[0].wq.forward(input, output, context);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 8.0f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 18.0f);
}

TEST(ModelStructureTest, BlockNormsUseTinyLlamaEpsilon) {
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

    model::TinyLlamaBlock block;
    for (auto* norm : {&block.attention_norm, &block.ffn_norm}) {
        norm->reset_param_size(1);
        norm->get_param(0)._data = weight;
        const auto status = norm->forward(input, output, context);
        ASSERT_TRUE(status.ok()) << status.message();
        // Small inputs distinguish TinyLlama's 1e-5 from the Operator default.
        const float inverse_rms = 1.0f / std::sqrt(2.5e-6f + 1e-5f);
        EXPECT_NEAR(output.ptr<float>()[0], 0.001f * inverse_rms, 1e-6f);
        EXPECT_NEAR(output.ptr<float>()[1], 0.002f * inverse_rms, 1e-6f);
    }
}
