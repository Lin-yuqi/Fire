#include <Fire/tokenizer/bpe.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace token {

size_t BPE::PairHash::operator()(const Pair& pair) const {
    size_t h1 = std::hash<std::string>{}(pair.left);
    size_t h2 = std::hash<std::string>{}(pair.right);

    return h1 ^ (h2 << 1);
}

BPE::BPE(MergeRanks ranks) : _merge_ranks(std::move(ranks)) {}

std::vector<BPE::Symbol> BPE::merge(const std::vector<BPE::Symbol>& symbols) const {
    if (symbols.size() < 2)
        return symbols;

    std::vector<Symbol> cur = symbols;
    while (cur.size() >= 2) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        Pair best_pair;
        bool found = false;

        // 找当前所有相邻pair中最小的
        for (int32_t i = 0; i + 1 < cur.size(); i++) {
            Pair pair{cur[i], cur[i + 1]};
            auto it = _merge_ranks.find(pair);
            if (it == _merge_ranks.end())
                continue;
            if (!found || it->second < best_rank) {
                best_rank = it->second;
                best_pair = std::move(pair);
                found = true;
            }
        }
        if (!found) {
            break;
        }

        std::vector<Symbol> next;
        next.reserve(cur.size());
        for (int i = 0; i < cur.size();) {
            if (i + 1 < cur.size() && cur[i] == best_pair.left && cur[i + 1] == best_pair.right) {
                next.emplace_back(cur[i] + cur[i + 1]);
                i += 2;
            } else {
                next.emplace_back(std::move(cur[i]));
                ++i;
            }
        }
        cur = std::move(next);
    }
    return cur;
}

} // namespace token
