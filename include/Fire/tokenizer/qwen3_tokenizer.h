#pragma once

#include "Fire/tokenizer/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace token {

class Qwen3Tokenizer final : public Tokenizer {
  public:
    Qwen3Tokenizer();
    ~Qwen3Tokenizer() override;

    // Read tokenizer.json and tokenizer_config.json from a model directory.
    // A failed load preserves the previously loaded tokenizer.
    base::Status load(const std::string& model_directory);

    // Qwen3 has no tokenizer BOS. Through Tokenizer, pass add_bos=false
    // explicitly because that interface defaults to true.
    base::Status encode(const std::string& text, std::vector<int32_t>& ids,
                        bool add_bos = false, bool add_eos = false) const override;

    // Preserve all added tokens, including special tokens. Incomplete UTF-8
    // byte sequences are replaced with U+FFFD after concatenating token bytes.
    base::Status decode(const std::vector<int32_t>& ids, std::string& text) const override;

    // Includes added tokens; this differs from the padded model vocabulary.
    int32_t vocab_size() const noexcept;
    int32_t bos_id() const noexcept;
    int32_t eos_id() const noexcept;
    int32_t unk_id() const noexcept;
    int32_t pad_id() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace token
