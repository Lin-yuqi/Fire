#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qwen_chat {

struct Turn {
    std::string user;
    std::string assistant;
};

// The local Qwen3 no-tools template removes reasoning from old assistant
// messages. An unfinished thinking block is retained, as in the HF template.
inline std::string historical_answer(const std::string& answer) {
    const auto end = answer.rfind("</think>");
    if (end == std::string::npos) {
        return answer;
    }
    const auto content = answer.find_first_not_of('\n', end + 8);
    return content == std::string::npos ? "" : answer.substr(content);
}

inline std::string format_prompt(const std::vector<Turn>& history,
                                 const std::string& user_input, bool thinking) {
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    for (const auto& turn : history) {
        prompt += "<|im_start|>user\n" + turn.user + "<|im_end|>\n";
        prompt += "<|im_start|>assistant\n" + historical_answer(turn.assistant) + "<|im_end|>\n";
    }
    prompt += "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}

inline bool can_reuse_cache(const std::vector<int32_t>& cached,
                            const std::vector<int32_t>& prompt) {
    return cached.size() <= prompt.size() && std::equal(cached.begin(), cached.end(), prompt.begin());
}

inline size_t response_budget(size_t prompt_size, size_t capacity,
                               size_t max_new_tokens, size_t closing_size) {
    if (closing_size >= capacity || prompt_size >= capacity - closing_size) {
        return 0;
    }
    return std::min(max_new_tokens, capacity - closing_size - prompt_size);
}

// ByteLevel decoding may temporarily replace a split UTF-8 character at the
// end. Delay trailing replacements until another token or the final flush.
inline size_t stable_text_size(const std::string& decoded) {
    size_t end = decoded.size();
    while (end >= 3 && decoded.compare(end - 3, 3, "\xEF\xBF\xBD") == 0) {
        end -= 3;
    }
    return end;
}

inline bool is_stop_token(int32_t id, int32_t eos, int32_t end_of_text) {
    return id == eos || id == end_of_text;
}

} // namespace qwen_chat
