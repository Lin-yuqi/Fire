#include "Fire/tokenizer/qwen3_tokenizer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

class TemporaryModel {
  public:
    TemporaryModel() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("fire-qwen3-tokenizer-" + std::to_string(suffix));
        if (!std::filesystem::create_directory(path)) {
            throw std::runtime_error("cannot create temporary tokenizer directory");
        }
    }

    ~TemporaryModel() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }

    void write(const Json& tokenizer, const Json& config) const {
        std::ofstream(path / "tokenizer.json") << tokenizer.dump();
        std::ofstream(path / "tokenizer_config.json") << config.dump();
    }

    std::filesystem::path path;
};

class Qwen3TokenizerLoadedTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!std::filesystem::exists(std::filesystem::path(FIRE_QWEN3_TOKENIZER_DIRECTORY) /
                                     "tokenizer.json")) {
            GTEST_SKIP() << "local Qwen3 tokenizer is unavailable";
        }
        const auto status = tokenizer.load(FIRE_QWEN3_TOKENIZER_DIRECTORY);
        ASSERT_TRUE(status.ok()) << status.message();
    }

    token::Qwen3Tokenizer tokenizer;
};

TEST(Qwen3TokenizerTest, RejectsUseBeforeLoadAndMissingFiles) {
    token::Qwen3Tokenizer tokenizer;
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
    EXPECT_EQ(tokenizer.pad_id(), -1);
    std::vector<int32_t> ids{42};
    std::string decoded = "unchanged";
    EXPECT_EQ(tokenizer.encode("hello", ids).code(), base::InternalError);
    EXPECT_EQ(tokenizer.decode({42}, decoded).code(), base::InternalError);
    EXPECT_EQ(ids, (std::vector<int32_t>{42}));
    EXPECT_EQ(decoded, "unchanged");
    TemporaryModel missing;
    EXPECT_EQ(tokenizer.load(missing.path.string()).code(), base::ModelParseError);
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    std::ofstream(missing.path / "tokenizer.json") << "{invalid JSON";
    std::ofstream(missing.path / "tokenizer_config.json") << "{}";
    EXPECT_EQ(tokenizer.load(missing.path.string()).code(), base::ModelParseError);
}

TEST_F(Qwen3TokenizerLoadedTest, ReportsVocabularyAndUsesExplicitBosEosRules) {
    EXPECT_EQ(tokenizer.vocab_size(), 151669);
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), 151645);
    EXPECT_EQ(tokenizer.pad_id(), 151643);

    std::vector<int32_t> ids{42};
    ASSERT_TRUE(tokenizer.encode("Hello", ids).ok());
    EXPECT_EQ(ids, (std::vector<int32_t>{9707}));
    ASSERT_TRUE(tokenizer.encode("Hello", ids, false, true).ok());
    EXPECT_EQ(ids, (std::vector<int32_t>{9707, 151645}));
    EXPECT_EQ(tokenizer.encode("Hello", ids, true).code(), base::InvalidArgument);
    EXPECT_EQ(ids, (std::vector<int32_t>{9707, 151645}));

    const token::Tokenizer& interface = tokenizer;
    EXPECT_EQ(interface.encode("Hello", ids).code(), base::InvalidArgument);
    ASSERT_TRUE(interface.encode("Hello", ids, false).ok());
    EXPECT_EQ(ids, (std::vector<int32_t>{9707}));
    ASSERT_TRUE(tokenizer.encode("", ids).ok());
    EXPECT_TRUE(ids.empty());
    ASSERT_TRUE(tokenizer.encode("", ids, false, true).ok());
    EXPECT_EQ(ids, (std::vector<int32_t>{151645}));
}

TEST_F(Qwen3TokenizerLoadedTest, MatchesLocalHuggingFaceGoldenIdsAndDecodedText) {
    struct Case {
        std::string input;
        std::vector<int32_t> ids;
        std::string decoded;
    };
    // Generated with local Qwen3-0.6B AutoTokenizer, add_special_tokens=False,
    // decode(skip_special_tokens=False, clean_up_tokenization_spaces=False).
    const std::vector<Case> cases{
        {"Hello, Qwen3!", {9707, 11, 1207, 16948, 18, 0}, "Hello, Qwen3!"},
        {"你好，世界！", {108386, 3837, 99489, 6313}, "你好，世界！"},
        {"🙂🚀 café", {145080, 145836, 51950}, "🙂🚀 café"},
        {"cafe\u0301", {924, 58858}, "café"},
        {"e\u0301 é E\u0301", {963, 3958, 28024}, "é é É"},
        {"1234567890 2026", {16, 17, 18, 19, 20, 21, 22, 23, 24, 15, 220, 17, 15, 17, 21},
         "1234567890 2026"},
        {"  hello\tworld\r\n\n  ", {220, 23811, 76508, 80823, 256}, "  hello\tworld\r\n\n  "},
        {"I'm WE'RE can't she'd", {40, 2776, 19677, 94153, 646, 944, 1340, 4172},
         "I'm WE'RE can't she'd"},
        {"I'M WE'LL THEY'D you're I've", {40, 27603, 19677, 6, 4086, 62493, 27705, 498, 2299, 358, 3003},
         "I'M WE'LL THEY'D you're I've"},
        {"a\u00a0b\u2003c\u0085d", {64, 4102, 65, 378, 225, 66, 126, 227, 67},
         "a\u00a0b\u2003c\u0085d"},
        {"a\u2028b\u2029c", {64, 378, 101, 65, 378, 102, 66}, "a\u2028b\u2029c"},
        {"👩‍💻🇨🇳", {145233, 378, 235, 145851, 145793, 145754}, "👩‍💻🇨🇳"},
        {"ＡＢＣ １２３ Ⅷ ²", {133054, 139173, 134557, 220, 20109, 24918, 33517, 220, 70467,
                                100, 220, 29456}, "ＡＢＣ １２３ Ⅷ ²"},
        {"한글 한글", {23573, 83291, 61298, 83291}, "한글 한글"},
        {std::string("\0\1\v\f\37", 5), {188, 189, 199, 200, 219}, std::string("\0\1\v\f\37", 5)},
        {"<|im_start|>user\n你好<think>思考</think><|im_end|>",
         {151644, 872, 198, 108386, 151667, 104107, 151668, 151645},
         "<|im_start|>user\n你好<think>思考</think><|im_end|>"},
        {"<tool_call>{\"name\":\"x\"}</tool_call><tool_response>ok</tool_response>",
         {151657, 4913, 606, 3252, 87, 9207, 151658, 151665, 562, 151666},
         "<tool_call>{\"name\":\"x\"}</tool_call><tool_response>ok</tool_response>"},
        {"<|im_start|><|im_start|><think></think><|im_end|>",
         {151644, 151644, 151667, 151668, 151645}, "<|im_start|><|im_start|><think></think><|im_end|>"},
        {"a<|im_start|>b<|endoftext|>c", {64, 151644, 65, 151643, 66},
         "a<|im_start|>b<|endoftext|>c"},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.input);
        std::vector<int32_t> ids;
        const auto encode_status = tokenizer.encode(item.input, ids);
        ASSERT_TRUE(encode_status.ok()) << encode_status.message();
        EXPECT_EQ(ids, item.ids);
        std::string decoded;
        const auto decode_status = tokenizer.decode(item.ids, decoded);
        ASSERT_TRUE(decode_status.ok()) << decode_status.message();
        EXPECT_EQ(decoded, item.decoded);
    }
}

TEST_F(Qwen3TokenizerLoadedTest, ConcatenatesBytesBeforeUtf8ReplacementAndPreservesAddedTokens) {
    std::string decoded = "unchanged";
    ASSERT_TRUE(tokenizer.decode({126, 227, 378, 225}, decoded).ok());
    EXPECT_EQ(decoded, "\u0085\u2003");
    ASSERT_TRUE(tokenizer.decode({253, 240}, decoded).ok());
    EXPECT_EQ(decoded, "��");
    ASSERT_TRUE(tokenizer.decode({151643, 151644, 151645, 151667, 151668}, decoded).ok());
    EXPECT_EQ(decoded, "<|endoftext|><|im_start|><|im_end|><think></think>");
    ASSERT_TRUE(tokenizer.decode({}, decoded).ok());
    EXPECT_TRUE(decoded.empty());
}

TEST_F(Qwen3TokenizerLoadedTest, RejectsInvalidUtf8AndUnknownIdsWithoutChangingOutputs) {
    std::vector<int32_t> ids{42};
    for (const auto& input : {std::string("\xFF"), std::string("\xC0\xAF"), std::string("\xE2\x82")}) {
        EXPECT_EQ(tokenizer.encode(input, ids).code(), base::InvalidArgument);
        EXPECT_EQ(ids, (std::vector<int32_t>{42}));
    }
    std::string decoded = "unchanged";
    for (int32_t id : {-1, 151669, 151935}) {
        EXPECT_EQ(tokenizer.decode({9707, id}, decoded).code(), base::InvalidArgument);
        EXPECT_EQ(decoded, "unchanged");
    }
}

TEST_F(Qwen3TokenizerLoadedTest, FailedReloadPreservesLoadedTokenizer) {
    Json original;
    Json config;
    std::ifstream(std::filesystem::path(FIRE_QWEN3_TOKENIZER_DIRECTORY) / "tokenizer.json") >> original;
    std::ifstream(std::filesystem::path(FIRE_QWEN3_TOKENIZER_DIRECTORY) / "tokenizer_config.json") >> config;
    TemporaryModel model;
    for (int invalid_case = 0; invalid_case < 7; ++invalid_case) {
        SCOPED_TRACE(invalid_case);
        Json changed = original;
        switch (invalid_case) {
        case 0: changed["normalizer"]["type"] = "NFKC"; break;
        case 1: changed["model"]["vocab"]["!"] = -1; break;
        case 2: changed["model"]["merges"][0] = Json::array({"missing-symbol", "x"}); break;
        case 3: changed["added_tokens"][0]["normalized"] = true; break;
        case 4: changed["added_tokens"][1]["id"] = 151643; break;
        case 5: changed["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\w+"; break;
        case 6: changed["model"]["dropout"] = 0.1; break;
        }
        model.write(changed, config);
        EXPECT_EQ(tokenizer.load(model.path.string()).code(), base::ModelParseError);
        EXPECT_EQ(tokenizer.vocab_size(), 151669);
        std::vector<int32_t> ids;
        ASSERT_TRUE(tokenizer.encode("Hello", ids).ok());
        EXPECT_EQ(ids, (std::vector<int32_t>{9707}));
    }
    model.write(original, config);
    ASSERT_TRUE(tokenizer.load(model.path.string()).ok());
}

TEST_F(Qwen3TokenizerLoadedTest, LoadsAwqTokenizerWithTheSameTextEncoding) {
    if (!std::filesystem::exists(std::filesystem::path(FIRE_QWEN3_AWQ_TOKENIZER_DIRECTORY) /
                                 "tokenizer.json")) {
        GTEST_SKIP() << "local AWQ tokenizer is unavailable";
    }
    const auto status = tokenizer.load(FIRE_QWEN3_AWQ_TOKENIZER_DIRECTORY);
    ASSERT_TRUE(status.ok()) << status.message();
    std::vector<int32_t> ids;
    ASSERT_TRUE(tokenizer.encode("你好，世界！", ids).ok());
    EXPECT_EQ(ids, (std::vector<int32_t>{108386, 3837, 99489, 6313}));
}

} // namespace
