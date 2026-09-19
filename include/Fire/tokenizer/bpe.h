#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace token {

class BPE {
  public:
    using Symbol = std::string;

    struct Pair {
        Symbol left;
        Symbol right;

        bool operator==(const Pair& other) const {
            return left == other.left && right == other.right;
        }
    };

    struct PairHash {
        size_t operator()(const Pair& pair) const;
    };

    using MergeRanks = std::unordered_map<Pair, int32_t, PairHash>;

    explicit BPE(MergeRanks ranks);
    BPE() = default;
    std::vector<Symbol> merge(const std::vector<Symbol>& symbols) const;

  private:
    MergeRanks _merge_ranks;
};

} // namespace token