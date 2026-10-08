#include "Fire/tokenizer/bpe.h"

#include <gtest/gtest.h>

TEST(BPETest, PreservesEmptySingleAndUnmergeableSymbols) {
    token::BPE bpe;
    EXPECT_TRUE(bpe.merge({}).empty());
    EXPECT_EQ(bpe.merge({"a"}), (std::vector<std::string>{"a"}));
    EXPECT_EQ(bpe.merge({"a", "b"}), (std::vector<std::string>{"a", "b"}));
}

TEST(BPETest, UsesRankBeforePositionAndRechecksNewNeighbors) {
    token::BPE bpe({{{"a", "b"}, 10}, {{"b", "c"}, 1}, {{"a", "bc"}, 2}});
    EXPECT_EQ(bpe.merge({"a", "b", "c"}), (std::vector<std::string>{"abc"}));
}

TEST(BPETest, MergesRepeatedPairsWithoutOverlappingSymbols) {
    token::BPE bpe({{{"a", "a"}, 0}, {{"aa", "aa"}, 1}});
    EXPECT_EQ(bpe.merge({"a", "a", "a"}), (std::vector<std::string>{"aa", "a"}));
    EXPECT_EQ(bpe.merge({"a", "a", "a", "a"}), (std::vector<std::string>{"aaaa"}));
}
