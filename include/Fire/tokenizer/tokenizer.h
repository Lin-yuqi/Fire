#pragma once
#include <Fire/base/base.h>
#include <vector>
namespace token {
// tokenizer.h
class Tokenizer {
  public:
    virtual ~Tokenizer() = default;

    virtual base::Status encode(const std::string& text, std::vector<int32_t>& ids,
                                bool add_bos = true, bool add_eos = false) const = 0;

    virtual base::Status decode(const std::vector<int32_t>& ids, std::string& text) const = 0;
};
} // namespace token
