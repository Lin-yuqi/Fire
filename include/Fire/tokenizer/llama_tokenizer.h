#pragma once

#include "Fire/base/base.h"
#include "Fire/tokenizer/tokenizer.h"

#include <sentencepiece_processor.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace token {

class LlamaTokenizer final : public Tokenizer {
  public:
    LlamaTokenizer() = default;

    base::Status load(const std::string& tokenizer_path);

    base::Status encode(const std::string& text,
                        std::vector<int32_t>& ids,
                        bool add_bos = true,
                        bool add_eos = false) const override;

    base::Status decode(const std::vector<int32_t>& ids,
                        std::string& text) const override;

    int32_t vocab_size() const noexcept;

    int32_t bos_id() const noexcept;
    int32_t eos_id() const noexcept;
    int32_t unk_id() const noexcept;

  private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> _processor;
};

} // namespace token