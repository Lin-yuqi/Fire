#pragma once

#include "Fire/base/base.h"
#include "Fire/tokenizer/bpe.h"
#include "Fire/tokenizer/tokenizer.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace token {

class LlamaTokenizer final : public Tokenizer {
public:
    LlamaTokenizer() = default;

    base::Status load(const std::string& tokenizer_path);

    base::Status encode(
        const std::string& text,
        std::vector<int32_t>& ids,
        bool add_bos = true,
        bool add_eos = false) const override;

    base::Status decode(
        const std::vector<int32_t>& ids,
        std::string& text) const override;

    int32_t vocab_size() const noexcept;

    int32_t bos_id() const noexcept;
    int32_t eos_id() const noexcept;
    int32_t unk_id() const noexcept;

private:
    std::vector<std::string> preprocess(
        const std::string& text) const;

private:
    BPE _bpe;

    std::unordered_map<std::string, int32_t> _piece_to_id;
    std::vector<std::string> _id_to_piece;

    int32_t _unk_id = 0;
    int32_t _bos_id = 1;
    int32_t _eos_id = 2;
};

} // namespace tokenizer