#include "../../demo/qwen_chat_utils.h"
#include "Fire/tokenizer/qwen3_tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

TEST(QwenChatTest, FormatsNonThinkingPromptLikeLocalHuggingFaceTemplate) {
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    EXPECT_EQ(qwen_chat::format_prompt({}, "你好", false), expected);
}

TEST(QwenChatTest, ThinkingPromptLeavesGenerationToTheModel) {
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(qwen_chat::format_prompt({}, "你好", true), expected);
}

TEST(QwenChatTest, RemovesOldReasoningAndPreservesMessageBoundaries) {
    const std::vector<qwen_chat::Turn> history{{"你好", "<think>\n推理\n</think>\n\n你好！"}};
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n你好！<|im_end|>\n"
        "<|im_start|>user\n继续<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    EXPECT_EQ(qwen_chat::format_prompt(history, "继续", false), expected);
    EXPECT_EQ(qwen_chat::historical_answer("<think>unfinished"), "<think>unfinished");
    EXPECT_EQ(qwen_chat::historical_answer("</think>\n\n"), "");
    EXPECT_EQ(qwen_chat::historical_answer("\nplain answer"), "\nplain answer");
}

TEST(QwenChatTest, OnlyReusesAnUnchangedTokenPrefix) {
    EXPECT_TRUE(qwen_chat::can_reuse_cache({}, {1, 2}));
    EXPECT_TRUE(qwen_chat::can_reuse_cache({1, 2}, {1, 2, 3}));
    EXPECT_FALSE(qwen_chat::can_reuse_cache({1, 2, 3}, {1, 2}));
    EXPECT_FALSE(qwen_chat::can_reuse_cache({1, 2}, {1, 3, 4}));
}

TEST(QwenChatTest, ReservesBothAssistantEndAndNewlineWithoutUnsignedUnderflow) {
    EXPECT_EQ(qwen_chat::response_budget(24, 2048, 128, 2), 128);
    EXPECT_EQ(qwen_chat::response_budget(2045, 2048, 128, 2), 1);
    EXPECT_EQ(qwen_chat::response_budget(2046, 2048, 128, 2), 0);
    EXPECT_EQ(qwen_chat::response_budget(4096, 2048, 128, 2), 0);
    EXPECT_EQ(qwen_chat::response_budget(0, 1, 128, 2), 0);
}

TEST(QwenChatTest, DelaysOnlyTrailingUtf8ReplacementCharacters) {
    EXPECT_EQ(qwen_chat::stable_text_size("中文"), std::string("中文").size());
    EXPECT_EQ(qwen_chat::stable_text_size("中��"), std::string("中").size());
    EXPECT_EQ(qwen_chat::stable_text_size("�中"), std::string("�中").size());
    EXPECT_EQ(qwen_chat::stable_text_size("�"), 0);
    EXPECT_EQ(qwen_chat::stable_text_size(""), 0);
}

TEST(QwenChatTest, RecognizesBothGenerationEosIds) {
    EXPECT_TRUE(qwen_chat::is_stop_token(151645, 151645, 151643));
    EXPECT_TRUE(qwen_chat::is_stop_token(151643, 151645, 151643));
    EXPECT_FALSE(qwen_chat::is_stop_token(151668, 151645, 151643));
}

TEST(QwenChatTest, PromptEncodingMatchesHuggingFaceAndHistoryChangesInvalidateCache) {
    if (!std::filesystem::exists(std::filesystem::path(FIRE_QWEN3_TOKENIZER_DIRECTORY) / "tokenizer.json")) {
        GTEST_SKIP() << "local Qwen3 tokenizer is unavailable";
    }
    token::Qwen3Tokenizer tokenizer;
    const auto loaded = tokenizer.load(FIRE_QWEN3_TOKENIZER_DIRECTORY);
    ASSERT_TRUE(loaded.ok()) << loaded.message();
    std::vector<int32_t> first;
    ASSERT_TRUE(tokenizer.encode(qwen_chat::format_prompt({}, "你好", false), first).ok());
    EXPECT_EQ(first, (std::vector<int32_t>{151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
        151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198,
        151667, 271, 151668, 271}));

    // A non-thinking generation prefix is absent from historical assistants.
    first.insert(first.end(), {108386, 6313, 151645, 198});
    std::vector<int32_t> second;
    ASSERT_TRUE(tokenizer.encode(qwen_chat::format_prompt({{"你好", "你好！"}}, "继续", false), second).ok());
    EXPECT_FALSE(qwen_chat::can_reuse_cache(first, second));
    EXPECT_EQ(second, (std::vector<int32_t>{151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
        151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198,
        108386, 6313, 151645, 198, 151644, 872, 198, 100640, 151645, 198, 151644,
        77091, 198, 151667, 271, 151668, 271}));
}
