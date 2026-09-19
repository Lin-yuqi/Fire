#include "Fire/tokenizer/llama_tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

TEST(LlamaTokenizerTest, LoadsTinyLlamaAndRoundTripsText) {
    token::LlamaTokenizer tokenizer;

    const base::Status load_status = tokenizer.load(FIRE_TINYLLAMA_TOKENIZER_PATH);
    ASSERT_TRUE(load_status.ok()) << load_status.message();
    EXPECT_EQ(tokenizer.vocab_size(), 32000);
    EXPECT_EQ(tokenizer.unk_id(), 0);
    EXPECT_EQ(tokenizer.bos_id(), 1);
    EXPECT_EQ(tokenizer.eos_id(), 2);

    const std::string input = "Hello, TinyLlama!";
    const std::vector<int32_t> expected_ids = {
        1, 15043, 29892, 323, 4901, 29931, 29880, 3304, 29991, 2,
    };
    std::vector<int32_t> ids;

    const base::Status encode_status = tokenizer.encode(input, ids, true, true);
    ASSERT_TRUE(encode_status.ok()) << encode_status.message();
    EXPECT_EQ(ids, expected_ids);

    std::string decoded;
    const base::Status decode_status = tokenizer.decode(ids, decoded);
    ASSERT_TRUE(decode_status.ok()) << decode_status.message();
    EXPECT_EQ(decoded, input);
}
