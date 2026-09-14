#include "Fire/model/model.h"
#include "Fire/model/tinyllama.h"

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>
#include <vector>

static_assert(std::is_abstract_v<model::Model>);
static_assert(!std::is_copy_constructible_v<model::TinyLlamaBlock>);
static_assert(std::is_move_constructible_v<model::TinyLlamaBlock>);
static_assert(std::is_move_assignable_v<model::TinyLlamaBlock>);

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
