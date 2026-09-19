#include "Fire/base/base.h"
#include <Fire/tokenizer/llama_tokenizer.h>
#include <memory>
#include <sentencepiece_processor.h>
#include <utility>

namespace token {
base::Status LlamaTokenizer::load(const std::string& tokenizer_path) {
    auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto status = processor->Load(tokenizer_path);
    if (!status.ok()) {
        return base::error::ModelParseError("failed to load sentencepiece tokenizer: " +
                                            status.ToString());
    }

    _processor = std::move(processor);
    return base::error::Success();
}
base::Status LlamaTokenizer::encode(const std::string& text, std::vector<int32_t>& ids,
                                    bool add_bos, bool add_eos) const {
    if (_processor == nullptr)
        return base::error::InternalError("tokenizer has not been loaded");

    ids.clear();
    auto status = _processor->Encode(text, &ids);
    if (!status.ok()) {
        return base::error::InternalError("sentencepiece encode failed: " + status.ToString());
    }

    if (add_bos) {
        ids.insert(ids.begin(), _processor->bos_id());
    }

    if (add_eos) {
        ids.push_back(_processor->eos_id());
    }

    return base::error::Success();
}

base::Status LlamaTokenizer::decode(const std::vector<int32_t>& ids, std::string& text) const {
    if (_processor == nullptr) {
        return base::error::InternalError("tokenizer has not been loaded");
    }

    auto status = _processor->Decode(ids, &text);
    if (!status.ok()) {
        return base::error::InternalError("sentencepiece decode failed: " + status.ToString());
    }

    return base::error::Success();
}

int32_t LlamaTokenizer::vocab_size() const noexcept {
    if (_processor == nullptr) {
        return 0;
    }
    return _processor->GetPieceSize();
}

int32_t LlamaTokenizer::bos_id() const noexcept { return _processor ? _processor->bos_id() : -1; }

int32_t LlamaTokenizer::eos_id() const noexcept { return _processor ? _processor->eos_id() : -1; }

int32_t LlamaTokenizer::unk_id() const noexcept { return _processor ? _processor->unk_id() : -1; }

} // namespace token