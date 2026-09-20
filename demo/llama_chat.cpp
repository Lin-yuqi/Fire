#include "Fire/base/alloc.h"
#include "Fire/model/model_weights.h"
#include "Fire/model/tinyllama.h"
#include "Fire/model/tinyllama_loader.h"
#include "Fire/sampler/argmax_sampler.h"
#include "Fire/tensor/tensor.h"
#include "Fire/tokenizer/llama_tokenizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr size_t MaxNewTokens = 128;
constexpr std::string_view SystemPrompt = "You are a helpful assistant.";
using PerformanceClock = std::chrono::steady_clock;

struct GenerationMetrics {
    PerformanceClock::duration time_to_first_token{};
    PerformanceClock::duration decode_time{};
    size_t sampled_tokens = 0;
};

class StreamCompletionGuard {
  public:
    explicit StreamCompletionGuard(cudaStream_t stream) : _stream(stream) {}

    ~StreamCompletionGuard() {
        if (_stream != nullptr) {
            cudaStreamSynchronize(_stream);
        }
    }

    StreamCompletionGuard(const StreamCompletionGuard&) = delete;
    StreamCompletionGuard& operator=(const StreamCompletionGuard&) = delete;

  private:
    cudaStream_t _stream = nullptr;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [model.fire] [tokenizer.model] [cpu|gpu]\n"
              << "Defaults:\n"
              << "  model:     " << FIRE_DEMO_MODEL_PATH << "\n"
              << "  tokenizer: " << FIRE_DEMO_TOKENIZER_PATH << "\n"
              << "  device:    gpu\n";
}

int report_error(std::string_view operation, const base::Status& status) {
    std::cerr << operation << " failed: " << status.message() << '\n';
    return 1;
}

void print_generation_metrics(const GenerationMetrics& metrics) {
    const double ttft_ms =
        std::chrono::duration<double, std::milli>(metrics.time_to_first_token).count();

    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "[performance] TTFT: " << ttft_ms
           << " ms, TPOT: ";

    const size_t decode_tokens = metrics.sampled_tokens > 0 ? metrics.sampled_tokens - 1 : 0;
    const double decode_seconds = std::chrono::duration<double>(metrics.decode_time).count();
    if (decode_tokens == 0 || decode_seconds <= 0.0) {
        output << "n/a, Decode throughput: n/a";
    } else {
        const double tpot_ms = decode_seconds * 1000.0 / static_cast<double>(decode_tokens);
        const double throughput = static_cast<double>(decode_tokens) / decode_seconds;
        output << tpot_ms << " ms/token, Decode throughput: " << throughput << " tokens/s";
    }
    std::cout << output.str() << '\n';
}

base::Status load_model(const std::string& model_path, const op::OpContext& context,
                        std::unique_ptr<model::TinyLlamaModel>& output) {
    model::TinyllamaLoader loader;
    auto status = loader.open(model_path);
    if (!status.ok()) {
        return status;
    }

    model::TinyLlamaWeights weights;
    status = loader.load_weights(weights);
    if (!status.ok()) {
        return status;
    }

    return model::TinyLlamaModel::create(weights, context, output);
}

base::Status append_encoded(const token::LlamaTokenizer& tokenizer, const std::string& text,
                            bool add_bos, std::vector<int32_t>& output) {
    std::vector<int32_t> ids;
    auto status = tokenizer.encode(text, ids, add_bos, false);
    if (!status.ok()) {
        return status;
    }
    output.insert(output.end(), ids.begin(), ids.end());
    return base::error::Success();
}

base::Status encode_user_turn(const token::LlamaTokenizer& tokenizer,
                              const std::string& user_input, bool first_turn,
                              std::vector<int32_t>& output) {
    output.clear();

    if (first_turn) {
        auto status = append_encoded(
            tokenizer, "<|system|>\n" + std::string(SystemPrompt), true, output);
        if (!status.ok()) {
            return status;
        }
        output.push_back(tokenizer.eos_id());
    }

    auto status = append_encoded(tokenizer, "\n<|user|>\n" + user_input, false, output);
    if (!status.ok()) {
        return status;
    }
    output.push_back(tokenizer.eos_id());
    return append_encoded(tokenizer, "\n<|assistant|>\n", false, output);
}

base::Status forward_tokens(model::TinyLlamaModel& model,
                            const std::vector<int32_t>& token_ids, tensor::Tensor& logits,
                            const op::OpContext& context, int32_t& position) {
    for (const int32_t token_id : token_ids) {
        auto status = model.forward(token_id, position, logits, context);
        if (!status.ok()) {
            return status;
        }
        ++position;
    }
    return base::error::Success();
}

base::Status generate_response(model::TinyLlamaModel& model,
                               const token::LlamaTokenizer& tokenizer,
                               sampler::ArgmaxSampler& sampler, size_t max_new_tokens,
                               tensor::Tensor& logits, const op::OpContext& context,
                               int32_t& position, PerformanceClock::time_point turn_started_at,
                               GenerationMetrics& metrics) {
    std::vector<int32_t> generated_ids;
    generated_ids.reserve(max_new_tokens);
    std::string printed_response;
    bool sampled_eos = false;
    bool decode_step_pending = false;
    PerformanceClock::time_point decode_step_started_at;

    for (size_t step = 0; step < max_new_tokens; ++step) {
        const size_t sampled_id =
            sampler.sample(logits.ptr<float>(), logits.size(), context._stream);
        const auto sample_completed_at = PerformanceClock::now();
        if (sampled_id >= static_cast<size_t>(model.config().vocab_size)) {
            return base::error::InternalError("sampler returned an invalid token id");
        }

        if (metrics.sampled_tokens == 0) {
            metrics.time_to_first_token = sample_completed_at - turn_started_at;
        } else if (decode_step_pending) {
            metrics.decode_time += sample_completed_at - decode_step_started_at;
        }
        ++metrics.sampled_tokens;

        if (sampled_id == static_cast<size_t>(tokenizer.eos_id())) {
            sampled_eos = true;
            break;
        }

        generated_ids.push_back(static_cast<int32_t>(sampled_id));

        std::string decoded;
        auto status = tokenizer.decode(generated_ids, decoded);
        if (!status.ok()) {
            return status;
        }
        if (decoded.compare(0, printed_response.size(), printed_response) == 0) {
            std::cout << decoded.substr(printed_response.size()) << std::flush;
            printed_response = std::move(decoded);
        }

        decode_step_started_at = PerformanceClock::now();
        decode_step_pending = true;
        status = model.forward(static_cast<int32_t>(sampled_id), position, logits, context);
        if (!status.ok()) {
            return status;
        }
        ++position;
    }

    if (!generated_ids.empty()) {
        std::string response;
        auto status = tokenizer.decode(generated_ids, response);
        if (!status.ok()) {
            return status;
        }
        if (response.compare(0, printed_response.size(), printed_response) == 0) {
            std::cout << response.substr(printed_response.size()) << std::flush;
        }
    }

    // Keep the cached token sequence equivalent to the chat template. If the
    // generation limit was reached, terminate the assistant turn explicitly.
    const int32_t eos_id = tokenizer.eos_id();
    auto status = model.forward(eos_id, position, logits, context);
    if (!status.ok()) {
        return status;
    }
    ++position;

    if (context._device_type == base::DeviceType::GPU) {
        const cudaError_t error = cudaStreamSynchronize(context._stream);
        if (error != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(error));
        }
    }

    if (!sampled_eos) {
        std::cout << "\n[response stopped after " << max_new_tokens << " tokens]";
    }
    return base::error::Success();
}

int run_chat(const std::string& model_path, const std::string& tokenizer_path,
             const op::OpContext& context) {
    token::LlamaTokenizer tokenizer;
    auto status = tokenizer.load(tokenizer_path);
    if (!status.ok()) {
        return report_error("loading tokenizer", status);
    }

    std::cout << "Loading TinyLlama model..." << std::flush;
    std::unique_ptr<model::TinyLlamaModel> model;
    status = load_model(model_path, context, model);
    if (!status.ok()) {
        std::cout << '\n';
        return report_error("loading model", status);
    }
    std::cout << " done\n";

    tensor::Tensor logits(base::DataType::Fp32, {model->config().vocab_size},
                          context._allocator);
    // Declared after GPU-backed objects: it synchronizes before they release
    // storage on every return path below.
    StreamCompletionGuard completion_guard(context._stream);

    status = model->prepare(model->config().max_seq_len, context);
    if (!status.ok()) {
        return report_error("preparing model", status);
    }

    sampler::ArgmaxSampler sampler(context._device_type);
    int32_t position = 0;
    bool first_turn = true;
    const size_t context_limit = static_cast<size_t>(model->config().max_seq_len);

    std::cout << "Fire TinyLlama chat (greedy, max " << MaxNewTokens
              << " new tokens, " << context_limit << " token context)\n"
              << "Commands: /reset, /exit, /help\n";

    while (true) {
        std::cout << "\nYou> " << std::flush;
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            std::cout << '\n';
            break;
        }
        if (user_input.empty()) {
            continue;
        }
        if (user_input == "/exit" || user_input == "/quit") {
            break;
        }
        if (user_input == "/help") {
            std::cout << "/reset starts a new conversation; /exit quits.\n";
            continue;
        }
        if (user_input == "/reset") {
            status = model->reset(context);
            if (!status.ok()) {
                return report_error("resetting model", status);
            }
            position = 0;
            first_turn = true;
            std::cout << "Conversation cleared.\n";
            continue;
        }

        std::vector<int32_t> turn_ids;
        const auto turn_started_at = PerformanceClock::now();
        status = encode_user_turn(tokenizer, user_input, first_turn, turn_ids);
        if (!status.ok()) {
            return report_error("encoding prompt", status);
        }

        const size_t used_tokens = static_cast<size_t>(position);
        const size_t remaining_tokens = context_limit - used_tokens;
        // Keep one slot for a generated token and one for the assistant EOS.
        if (turn_ids.size() + 2 > remaining_tokens) {
            std::cout << "Context window has only " << remaining_tokens
                      << " token(s) left; this message needs " << turn_ids.size() + 2
                      << ". Enter a shorter message, /reset, or /exit.\n";
            continue;
        }

        status = forward_tokens(*model, turn_ids, logits, context, position);
        if (!status.ok()) {
            return report_error("prefilling user turn", status);
        }
        first_turn = false;

        const size_t response_budget =
            std::min(MaxNewTokens, context_limit - static_cast<size_t>(position) - 1);

        std::cout << "Assistant> " << std::flush;
        GenerationMetrics metrics;
        status = generate_response(*model, tokenizer, sampler, response_budget, logits,
                                   context, position, turn_started_at, metrics);
        std::cout << '\n';
        if (!status.ok()) {
            return report_error("generating response", status);
        }
        print_generation_metrics(metrics);

        const size_t tokens_left = context_limit - static_cast<size_t>(position);
        if (tokens_left == 0) {
            std::cout << "Context window reached " << context_limit
                      << " tokens. Ending this chat session.\n";
            break;
        }
        std::cout << "[context: " << position << '/' << context_limit << " tokens used]\n";
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 0;
    }
    if (argc > 4) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string model_path = argc > 1 ? argv[1] : FIRE_DEMO_MODEL_PATH;
    const std::string tokenizer_path = argc > 2 ? argv[2] : FIRE_DEMO_TOKENIZER_PATH;
    const std::string device_name = argc > 3 ? argv[3] : "gpu";

    op::OpContext context;
    cudaStream_t stream = nullptr;
    if (device_name == "cpu") {
        context._device_type = base::DeviceType::CPU;
        context._allocator = base::CPUAllocatorFactory::get_instance();
    } else if (device_name == "gpu" || device_name == "cuda") {
        int device_count = 0;
        const cudaError_t device_error = cudaGetDeviceCount(&device_count);
        if (device_error != cudaSuccess || device_count == 0) {
            std::cerr << "CUDA device unavailable: " << cudaGetErrorString(device_error) << '\n';
            return 1;
        }
        const cudaError_t stream_error =
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_error != cudaSuccess) {
            std::cerr << "Creating CUDA stream failed: " << cudaGetErrorString(stream_error)
                      << '\n';
            return 1;
        }
        context._device_type = base::DeviceType::GPU;
        context._allocator = base::GPUAllocatorFactory::get_instance();
        context._stream = stream;
    } else {
        std::cerr << "Unknown device '" << device_name << "'. Expected cpu or gpu.\n";
        print_usage(argv[0]);
        return 2;
    }

    const int result = run_chat(model_path, tokenizer_path, context);
    if (stream != nullptr) {
        const cudaError_t error = cudaStreamDestroy(stream);
        if (error != cudaSuccess) {
            std::cerr << "Destroying CUDA stream failed: " << cudaGetErrorString(error) << '\n';
            return 1;
        }
    }
    return result;
}
