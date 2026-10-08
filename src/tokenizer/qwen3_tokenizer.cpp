#include "Fire/tokenizer/qwen3_tokenizer.h"

#include "Fire/tokenizer/bpe.h"

#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/stringpiece.h>
#include <unicode/unistr.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace token {
namespace {

using Json = nlohmann::json;

constexpr char QwenSplitPattern[] =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Json read_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(input.is_open(), "cannot open " + path.string());
    return Json::parse(input);
}

int32_t token_id(const Json& value) {
    require(value.is_number_integer(), "token ID must be an integer");
    const auto id = value.get<int64_t>();
    require(id >= 0 && id <= std::numeric_limits<int32_t>::max(), "token ID out of range");
    return static_cast<int32_t>(id);
}

bool valid_utf8(const std::string& text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
    const int32_t length = static_cast<int32_t>(text.size());
    for (int32_t index = 0; index < length;) {
        UChar32 codepoint;
        U8_NEXT(bytes, index, length, codepoint);
        if (codepoint < 0) {
            return false;
        }
    }
    return true;
}

void validate_byte_level(const Json& value) {
    require(value.at("type") == "ByteLevel" &&
                !value.at("add_prefix_space").get<bool>() &&
                !value.at("trim_offsets").get<bool>() && !value.at("use_regex").get<bool>(),
            "unsupported ByteLevel settings");
}

void validate_pipeline(const Json& tokenizer) {
    require(tokenizer.at("normalizer").at("type") == "NFC", "expected NFC normalization");
    const auto& pre = tokenizer.at("pre_tokenizer");
    const auto& stages = pre.at("pretokenizers");
    require(pre.at("type") == "Sequence" && stages.is_array() && stages.size() == 2,
            "expected Split then ByteLevel pre-tokenizers");
    const auto& split = stages.at(0);
    require(split.at("type") == "Split" && split.at("behavior") == "Isolated" &&
                !split.at("invert").get<bool>() &&
                split.at("pattern").at("Regex") == QwenSplitPattern,
            "unsupported Qwen3 split pattern");
    validate_byte_level(stages.at(1));
    validate_byte_level(tokenizer.at("post_processor"));
    validate_byte_level(tokenizer.at("decoder"));

    const auto& model = tokenizer.at("model");
    require(model.at("type") == "BPE" && model.at("dropout").is_null() &&
                model.at("unk_token").is_null() && model.at("continuing_subword_prefix") == "" &&
                model.at("end_of_word_suffix") == "" && !model.at("fuse_unk").get<bool>() &&
                !model.at("byte_fallback").get<bool>() && !model.at("ignore_merges").get<bool>(),
            "unsupported BPE settings");
}

} // namespace

struct Qwen3Tokenizer::Impl {
    struct AddedToken {
        std::string content;
        int32_t id;
    };

    BPE bpe;
    std::unordered_map<std::string, int32_t> vocabulary;
    // Base entries contain decoded bytes; added entries contain literal text.
    std::vector<std::string> token_bytes;
    std::vector<AddedToken> added_tokens;
    std::array<std::string, 256> byte_symbols;
    std::unique_ptr<icu::RegexPattern> split_pattern;
    const icu::Normalizer2* normalizer = nullptr; // ICU owns this singleton.
    int32_t eos = -1;
    int32_t pad = -1;

    void load(const Json& tokenizer, const Json& config) {
        validate_pipeline(tokenizer);
        require(config.at("bos_token").is_null() && !config.at("add_bos_token").get<bool>() &&
                    config.at("unk_token").is_null() && !config.at("add_prefix_space").get<bool>() &&
                    !config.at("clean_up_tokenization_spaces").get<bool>() &&
                    !config.at("split_special_tokens").get<bool>() && config.at("errors") == "replace",
                "unsupported tokenizer configuration");

        // GPT-2 ByteLevel mapping: visible bytes keep their code points;
        // remaining bytes map to consecutive code points starting at 256.
        std::unordered_map<UChar32, uint8_t> symbol_bytes;
        UChar32 extra = 256;
        for (int byte = 0; byte < 256; ++byte) {
            const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                                 byte >= 174;
            const UChar32 codepoint = visible ? byte : extra++;
            icu::UnicodeString(codepoint).toUTF8String(byte_symbols[byte]);
            symbol_bytes.emplace(codepoint, static_cast<uint8_t>(byte));
        }

        const auto& model = tokenizer.at("model");
        const auto& vocab = model.at("vocab");
        const auto& added = tokenizer.at("added_tokens");
        require(vocab.is_object() && vocab.size() >= 256 && added.is_array(), "invalid vocabulary");
        const size_t total = vocab.size() + added.size();
        require(total <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "vocabulary is too large");
        token_bytes.resize(total);
        std::vector<bool> assigned(total, false);
        vocabulary.reserve(vocab.size());
        for (const auto& entry : vocab.items()) {
            const int32_t id = token_id(entry.value());
            require(static_cast<size_t>(id) < vocab.size() && !assigned[id],
                    "base vocabulary IDs must be unique and contiguous");
            require(!entry.key().empty() && valid_utf8(entry.key()), "invalid vocabulary symbol");
            const auto symbols = icu::UnicodeString::fromUTF8(entry.key());
            std::string bytes;
            for (int32_t index = 0; index < symbols.length();) {
                const UChar32 codepoint = symbols.char32At(index);
                index += U16_LENGTH(codepoint);
                const auto it = symbol_bytes.find(codepoint);
                require(it != symbol_bytes.end(), "vocabulary symbol is outside ByteLevel alphabet");
                bytes.push_back(static_cast<char>(it->second));
            }
            token_bytes[id] = std::move(bytes);
            vocabulary.emplace(entry.key(), id);
            assigned[id] = true;
        }
        for (const auto& symbol : byte_symbols) {
            require(vocabulary.count(symbol) == 1, "missing byte symbol in vocabulary");
        }

        const auto& merges = model.at("merges");
        require(merges.is_array() &&
                    merges.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                "invalid merge ranks");
        BPE::MergeRanks ranks;
        ranks.reserve(merges.size());
        int32_t rank = 0;
        for (const auto& merge : merges) {
            BPE::Pair pair;
            if (merge.is_array() && merge.size() == 2) {
                pair = {merge.at(0).get<std::string>(), merge.at(1).get<std::string>()};
            } else if (merge.is_string()) {
                const auto value = merge.get<std::string>();
                const auto separator = value.find(' ');
                require(separator != std::string::npos, "invalid merge pair");
                pair = {value.substr(0, separator), value.substr(separator + 1)};
            } else {
                throw std::runtime_error("invalid merge pair");
            }
            require(vocabulary.count(pair.left) && vocabulary.count(pair.right) &&
                        vocabulary.count(pair.left + pair.right),
                    "merge references missing vocabulary symbol");
            require(ranks.emplace(std::move(pair), rank++).second, "duplicate merge pair");
        }
        bpe = BPE(std::move(ranks));

        std::unordered_map<std::string, int32_t> added_ids;
        for (const auto& value : added) {
            AddedToken entry{value.at("content").get<std::string>(), token_id(value.at("id"))};
            require(!entry.content.empty() && valid_utf8(entry.content) &&
                        static_cast<size_t>(entry.id) >= vocab.size() &&
                        static_cast<size_t>(entry.id) < total && !assigned[entry.id],
                    "invalid added token content or ID");
            require(!value.at("single_word").get<bool>() && !value.at("lstrip").get<bool>() &&
                        !value.at("rstrip").get<bool>() && !value.at("normalized").get<bool>() &&
                        value.at("special").is_boolean(),
                    "unsupported added token matching settings");
            require(!vocabulary.count(entry.content) &&
                        added_ids.emplace(entry.content, entry.id).second,
                    "duplicate added token");
            token_bytes[entry.id] = entry.content;
            assigned[entry.id] = true;
            added_tokens.push_back(std::move(entry));
        }
        require(std::all_of(assigned.begin(), assigned.end(), [](bool value) { return value; }),
                "vocabulary IDs must be contiguous");
        require(config.at("eos_token") == "<|im_end|>" &&
                    config.at("pad_token") == "<|endoftext|>" &&
                    added_ids.count("<|im_start|>") && added_ids.count("<|im_end|>") &&
                    added_ids.count("<|endoftext|>"),
                "missing Qwen3 control tokens");
        eos = added_ids.at("<|im_end|>");
        pad = added_ids.at("<|endoftext|>");

        UErrorCode error = U_ZERO_ERROR;
        normalizer = icu::Normalizer2::getNFCInstance(error);
        require(U_SUCCESS(error) && normalizer != nullptr, "cannot initialize NFC normalizer");
        split_pattern.reset(icu::RegexPattern::compile(
            icu::UnicodeString::fromUTF8(QwenSplitPattern), 0, error));
        require(U_SUCCESS(error) && split_pattern != nullptr, "cannot compile Qwen3 split pattern");
    }

    base::Status encode_ordinary(const std::string& text, std::vector<int32_t>& ids) const {
        UErrorCode error = U_ZERO_ERROR;
        icu::UnicodeString normalized;
        normalizer->normalize(icu::UnicodeString::fromUTF8(text), normalized, error);
        if (U_FAILURE(error)) {
            return base::error::InternalError("NFC normalization failed");
        }
        auto matcher = std::unique_ptr<icu::RegexMatcher>(split_pattern->matcher(normalized, error));
        if (U_FAILURE(error) || matcher == nullptr) {
            return base::error::InternalError("cannot create pre-tokenizer matcher");
        }
        int32_t consumed = 0;
        while (matcher->find(error)) {
            const int32_t start = matcher->start(error);
            const int32_t end = matcher->end(error);
            if (U_FAILURE(error) || start != consumed || end <= start) {
                return base::error::InternalError("pre-tokenizer did not cover input");
            }
            consumed = end;
            std::string bytes;
            normalized.tempSubStringBetween(start, end).toUTF8String(bytes);
            std::vector<BPE::Symbol> symbols;
            symbols.reserve(bytes.size());
            for (unsigned char byte : bytes) {
                symbols.push_back(byte_symbols[byte]);
            }
            for (const auto& symbol : bpe.merge(symbols)) {
                const auto it = vocabulary.find(symbol);
                if (it == vocabulary.end()) {
                    return base::error::InternalError("BPE produced an unknown symbol");
                }
                ids.push_back(it->second);
            }
        }
        if (U_FAILURE(error) || consumed != normalized.length()) {
            return base::error::InternalError("pre-tokenizer did not cover input");
        }
        return base::error::Success();
    }
};

Qwen3Tokenizer::Qwen3Tokenizer() = default;
Qwen3Tokenizer::~Qwen3Tokenizer() = default;

base::Status Qwen3Tokenizer::load(const std::string& model_directory) {
    try {
        const std::filesystem::path root(model_directory);
        auto impl = std::make_unique<Impl>();
        impl->load(read_json(root / "tokenizer.json"), read_json(root / "tokenizer_config.json"));
        _impl = std::move(impl);
        return base::error::Success();
    } catch (const std::exception& error) {
        return base::error::ModelParseError("failed to load Qwen3 tokenizer: " +
                                            std::string(error.what()));
    }
}

base::Status Qwen3Tokenizer::encode(const std::string& text, std::vector<int32_t>& ids,
                                    bool add_bos, bool add_eos) const {
    if (!_impl) {
        return base::error::InternalError("tokenizer has not been loaded");
    }
    if (add_bos) {
        return base::error::InvalidArgument("Qwen3 tokenizer has no BOS; pass add_bos=false");
    }
    if (!valid_utf8(text)) {
        return base::error::InvalidArgument("input must be valid UTF-8 within INT32_MAX bytes");
    }
    std::vector<int32_t> encoded;
    size_t position = 0;
    while (position < text.size()) {
        size_t next = text.size();
        const Impl::AddedToken* match = nullptr;
        for (const auto& added : _impl->added_tokens) {
            const auto found = text.find(added.content, position);
            if (found != std::string::npos &&
                (found < next ||
                 (found == next && match && added.content.size() > match->content.size()))) {
                next = found;
                match = &added;
            }
        }
        if (next > position) {
            auto status = _impl->encode_ordinary(text.substr(position, next - position), encoded);
            if (!status) {
                return status;
            }
        }
        if (!match) {
            break;
        }
        encoded.push_back(match->id);
        position = next + match->content.size();
    }
    if (add_eos) {
        encoded.push_back(_impl->eos);
    }
    ids = std::move(encoded);
    return base::error::Success();
}

base::Status Qwen3Tokenizer::decode(const std::vector<int32_t>& ids, std::string& text) const {
    if (!_impl) {
        return base::error::InternalError("tokenizer has not been loaded");
    }
    std::string bytes;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= _impl->token_bytes.size()) {
            return base::error::InvalidArgument("token ID is outside tokenizer vocabulary");
        }
        bytes += _impl->token_bytes[id];
    }
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return base::error::InvalidArgument("decoded text exceeds INT32_MAX bytes");
    }
    std::string decoded;
    icu::UnicodeString::fromUTF8(bytes).toUTF8String(decoded);
    text = std::move(decoded);
    return base::error::Success();
}

int32_t Qwen3Tokenizer::vocab_size() const noexcept {
    return _impl ? static_cast<int32_t>(_impl->token_bytes.size()) : 0;
}
int32_t Qwen3Tokenizer::bos_id() const noexcept { return -1; }
int32_t Qwen3Tokenizer::eos_id() const noexcept { return _impl ? _impl->eos : -1; }
int32_t Qwen3Tokenizer::unk_id() const noexcept { return -1; }
int32_t Qwen3Tokenizer::pad_id() const noexcept { return _impl ? _impl->pad : -1; }

} // namespace token
