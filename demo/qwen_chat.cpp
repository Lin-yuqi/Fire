#include "qwen_chat_utils.h"

#include "Fire/base/alloc.h"
#include "Fire/model/qwen.h"
#include "Fire/model/qwen_loader.h"
#include "Fire/sampler/argmax_sampler.h"
#include "Fire/tokenizer/qwen3_tokenizer.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

using PerformanceClock = std::chrono::steady_clock;

struct Options {
    const model::Qwen3Profile* profile = &model::qwen3_profiles::Qwen3_8B;
    std::string model_path;
    std::string tokenizer_directory;
    std::string device = "gpu";
    int32_t capacity = 2048;
    int32_t max_new_tokens = 128;
    bool thinking = false;
};

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
    cudaStream_t _stream;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [model.fire] [tokenizer-directory] [cpu|gpu]\n"
              << "  --profile 0.6b|8b      Model profile (default: 8b)\n"
              << "  --context-size N      KV capacity (default: 2048)\n"
              << "  --max-new-tokens N    Reply limit (default: 128)\n"
              << "  --thinking            Enable thinking (default: disabled)\n"
              << "  --help                Show usage\n"
              << "Defaults for 8b:\n"
              << "  model:     " << FIRE_QWEN3_8B_MODEL_PATH << '\n'
              << "  tokenizer: " << FIRE_QWEN3_8B_TOKENIZER_DIRECTORY << '\n'
              << "  device:    gpu\n"
              << "The 0.6b profile defaults to " << FIRE_QWEN3_0_6B_MODEL_PATH << '\n'
              << "and " << FIRE_QWEN3_0_6B_TOKENIZER_DIRECTORY << ".\n";
    std::cout << "CPU requires FP32 weights; INT4/AWQ models require GPU.\n";
}

bool parse_positive_integer(std::string_view text, int32_t& value) {
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size() && value > 0;
}

base::Status parse_options(int argc, char** argv, Options& options) {
    std::vector<std::string> positional;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--thinking") {
            options.thinking = true;
        } else if (argument == "--profile" || argument == "--context-size" ||
                   argument == "--max-new-tokens") {
            if (++index == argc) {
                return base::error::InvalidArgument(std::string(argument) + " requires a value");
            }
            const std::string_view value(argv[index]);
            if (argument == "--profile") {
                if (value == "0.6b") {
                    options.profile = &model::qwen3_profiles::Qwen3_0_6B;
                } else if (value == "8b") {
                    options.profile = &model::qwen3_profiles::Qwen3_8B;
                } else {
                    return base::error::InvalidArgument("profile must be 0.6b or 8b");
                }
            } else {
                int32_t& target =
                    argument == "--context-size" ? options.capacity : options.max_new_tokens;
                if (!parse_positive_integer(value, target)) {
                    return base::error::InvalidArgument(
                        std::string(argument) + " must be a positive integer");
                }
            }
        } else if (argument.size() >= 2 && argument.substr(0, 2) == "--") {
            return base::error::InvalidArgument("unknown option: " + std::string(argument));
        } else {
            positional.emplace_back(argument);
        }
    }
    if (positional.size() > 3) {
        return base::error::InvalidArgument("expected at most three positional arguments");
    }
    const bool small = options.profile == &model::qwen3_profiles::Qwen3_0_6B;
    options.model_path = positional.empty()
                             ? (small ? FIRE_QWEN3_0_6B_MODEL_PATH : FIRE_QWEN3_8B_MODEL_PATH)
                             : positional[0];
    options.tokenizer_directory = positional.size() < 2
                                     ? (small ? FIRE_QWEN3_0_6B_TOKENIZER_DIRECTORY
                                              : FIRE_QWEN3_8B_TOKENIZER_DIRECTORY)
                                     : positional[1];
    if (positional.size() == 3) {
        options.device = positional[2];
    }
    if (options.device != "cpu" && options.device != "gpu" && options.device != "cuda") {
        return base::error::InvalidArgument("device must be cpu or gpu");
    }
    if (options.capacity > options.profile->model.max_seq_len) {
        return base::error::InvalidArgument("context size exceeds the selected model profile");
    }
    return base::error::Success();
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
    const double seconds = std::chrono::duration<double>(metrics.decode_time).count();
    if (decode_tokens == 0 || seconds <= 0.0) {
        output << "n/a, Decode throughput: n/a";
    } else {
        output << seconds * 1000.0 / decode_tokens << " ms/token, Decode throughput: "
               << decode_tokens / seconds << " tokens/s";
    }
    std::cout << output.str() << '\n';
}

base::Status load_model(const Options& options, const op::OpContext& context,
                        std::unique_ptr<model::Qwen3Model>& output) {
    model::Qwen3Loader loader(*options.profile);
    auto status = loader.open(options.model_path);
    if (!status) {
        return status;
    }
    model::Qwen3Weights weights;
    status = loader.load_weights(weights);
    if (!status) {
        return status;
    }
    // The loader validates that every layer uses the same projection layout.
    if (context._device_type == base::DeviceType::CPU &&
        (weights.output.is_quantized() ||
         std::any_of(weights.layers.begin(), weights.layers.end(),
                     [](const model::Qwen3LayerWeights& layer) { return layer.wq.is_quantized(); }))) {
        return base::error::InvalidArgument("INT4/AWQ models require gpu; use FP32 weights for cpu");
    }
    if (context._device_type == base::DeviceType::GPU) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(error));
        }
        const auto& profile = *options.profile;
        const size_t kv_bytes =
            2ULL * profile.num_layers * options.capacity * profile.kv_dim() * sizeof(float);
        const size_t rope_bytes =
            static_cast<size_t>(profile.model.max_seq_len) * profile.head_dim * sizeof(float);
        std::error_code file_error;
        const auto weight_bytes = std::filesystem::file_size(options.model_path, file_error);
        if (file_error) {
            return base::error::PathNotValid(file_error.message());
        }
        if (free_bytes < weight_bytes + kv_bytes + rope_bytes + 128ULL * 1024 * 1024) {
            return base::error::InvalidArgument(
                "insufficient free GPU memory; reduce --context-size or select a smaller model");
        }
    }
    return model::Qwen3Model::create(weights, context, output);
}

base::Status forward_tokens(model::Qwen3Model& model, const std::vector<int32_t>& ids,
                            size_t offset, tensor::Tensor& logits, const op::OpContext& context,
                            std::vector<int32_t>& cached) {
    for (size_t index = offset; index < ids.size(); ++index) {
        auto status = model.forward(ids[index], static_cast<int32_t>(cached.size()), logits, context);
        if (!status) {
            return status;
        }
        cached.push_back(ids[index]);
    }
    return base::error::Success();
}

base::Status print_response(const token::Qwen3Tokenizer& tokenizer,
                            const std::vector<int32_t>& generated, bool final,
                            std::string& printed, std::string& response) {
    auto status = tokenizer.decode(generated, response);
    if (!status) {
        return status;
    }
    if (response.compare(0, printed.size(), printed) != 0) {
        return base::error::InternalError("decoded response changed its printed prefix");
    }
    const size_t end = final ? response.size() : qwen_chat::stable_text_size(response);
    std::cout << response.substr(printed.size(), end - printed.size()) << std::flush;
    printed.assign(response, 0, end);
    return base::error::Success();
}

base::Status generate_response(model::Qwen3Model& model, const token::Qwen3Tokenizer& tokenizer,
                               sampler::ArgmaxSampler& sampler, size_t budget,
                               const std::vector<int32_t>& closing, tensor::Tensor& logits,
                               const op::OpContext& context, std::vector<int32_t>& cached,
                               PerformanceClock::time_point started_at, GenerationMetrics& metrics,
                               std::string& response) {
    std::vector<int32_t> generated;
    generated.reserve(budget);
    std::string printed;
    bool stopped = false;
    PerformanceClock::time_point decode_started_at;
    for (size_t step = 0; step < budget; ++step) {
        // Model logits include padding rows without tokenizer entries.
        const size_t id =
            sampler.sample(logits.ptr<float>(), tokenizer.vocab_size(), context._stream);
        const auto sampled_at = PerformanceClock::now();
        if (id >= static_cast<size_t>(tokenizer.vocab_size())) {
            return base::error::InternalError("sampler returned an invalid token ID");
        }
        if (metrics.sampled_tokens == 0) {
            metrics.time_to_first_token = sampled_at - started_at;
        } else {
            metrics.decode_time += sampled_at - decode_started_at;
        }
        ++metrics.sampled_tokens;
        if (qwen_chat::is_stop_token(static_cast<int32_t>(id), tokenizer.eos_id(),
                                     tokenizer.pad_id())) {
            stopped = true;
            break;
        }
        generated.push_back(static_cast<int32_t>(id));
        auto status = print_response(tokenizer, generated, false, printed, response);
        if (!status) {
            return status;
        }
        decode_started_at = PerformanceClock::now();
        status = model.forward(static_cast<int32_t>(id), static_cast<int32_t>(cached.size()),
                               logits, context);
        if (!status) {
            return status;
        }
        cached.push_back(static_cast<int32_t>(id));
    }
    auto status = print_response(tokenizer, generated, true, printed, response);
    if (!status) {
        return status;
    }
    // Always close the assistant message with <|im_end|> plus newline, even
    // after endoftext or a token limit, so the next user message is well formed.
    status = forward_tokens(model, closing, 0, logits, context, cached);
    if (!status) {
        return status;
    }
    if (context._device_type == base::DeviceType::GPU) {
        const auto error = cudaStreamSynchronize(context._stream);
        if (error != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(error));
        }
    }
    if (!stopped) {
        std::cout << "\n[response stopped after " << budget << " tokens]";
    }
    return base::error::Success();
}

int run_chat(const Options& options, const op::OpContext& context) {
    token::Qwen3Tokenizer tokenizer;
    auto status = tokenizer.load(options.tokenizer_directory);
    if (!status) {
        return report_error("loading tokenizer", status);
    }
    if (tokenizer.vocab_size() > options.profile->model.vocab_size) {
        return report_error("loading tokenizer", base::error::InvalidArgument(
                                                     "tokenizer vocabulary exceeds model vocabulary"));
    }
    std::vector<int32_t> closing;
    status = tokenizer.encode("<|im_end|>\n", closing, false, false);
    if (!status) {
        return report_error("encoding message terminator", status);
    }

    std::cout << "Loading " << options.profile->model.profile_name << " model..." << std::flush;
    std::unique_ptr<model::Qwen3Model> model;
    status = load_model(options, context, model);
    if (!status) {
        std::cout << '\n';
        return report_error("loading model", status);
    }
    std::cout << " done\n";
    tensor::Tensor logits(base::DataType::Fp32, {model->config().vocab_size}, context._allocator);
    StreamCompletionGuard completion_guard(context._stream);
    status = model->prepare(options.capacity, context);
    if (!status) {
        return report_error("preparing model", status);
    }
    sampler::ArgmaxSampler sampler(context._device_type);
    std::vector<qwen_chat::Turn> history;
    std::vector<int32_t> cached;
    const size_t capacity = static_cast<size_t>(options.capacity);
    std::cout << "Fire v" << FIRE_VERSION << " Qwen3 chat (greedy, max " << options.max_new_tokens
              << " new tokens, " << capacity << " token context, thinking "
              << (options.thinking ? "enabled" : "disabled") << ")\n"
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
            if (!status) {
                return report_error("resetting model", status);
            }
            cached.clear();
            history.clear();
            std::cout << "Conversation cleared.\n";
            continue;
        }
        const auto started_at = PerformanceClock::now();
        std::vector<int32_t> prompt;
        status = tokenizer.encode(qwen_chat::format_prompt(history, user_input, options.thinking),
                                  prompt, false, false);
        if (!status) {
            return report_error("encoding prompt", status);
        }
        const size_t budget = qwen_chat::response_budget(
            prompt.size(), capacity, options.max_new_tokens, closing.size());
        if (budget == 0) {
            std::cout << "Message and history need at least " << prompt.size() + closing.size() + 1
                      << " tokens; context capacity is " << capacity
                      << ". Enter a shorter message, /reset, or /exit.\n";
            continue;
        }
        if (!qwen_chat::can_reuse_cache(cached, prompt)) {
            status = model->reset(context);
            if (!status) {
                return report_error("resetting model for history replay", status);
            }
            cached.clear();
        }
        status = forward_tokens(*model, prompt, cached.size(), logits, context, cached);
        if (!status) {
            return report_error("prefilling prompt", status);
        }
        std::cout << "Assistant> " << std::flush;
        GenerationMetrics metrics;
        std::string response;
        status = generate_response(*model, tokenizer, sampler, budget, closing, logits, context,
                                   cached, started_at, metrics, response);
        std::cout << '\n';
        if (!status) {
            return report_error("generating response", status);
        }
        history.push_back({std::move(user_input), std::move(response)});
        print_generation_metrics(metrics);
        std::cout << "[context: " << cached.size() << '/' << capacity << " tokens used]\n";
        if (cached.size() == capacity) {
            std::cout << "Context window reached " << capacity << " tokens. Enter /reset or /exit.\n";
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    Options options;
    auto status = parse_options(argc, argv, options);
    if (!status) {
        report_error("parsing arguments", status);
        print_usage(argv[0]);
        return 2;
    }
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = google::GLOG_WARNING;
    op::OpContext context;
    cudaStream_t stream = nullptr;
    if (options.device == "cpu") {
        context._device_type = base::DeviceType::CPU;
        context._allocator = base::CPUAllocatorFactory::get_instance();
    } else {
        int device_count = 0;
        const auto error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "CUDA device unavailable: "
                      << (error == cudaSuccess ? "no CUDA devices" : cudaGetErrorString(error)) << '\n';
            return 1;
        }
        const auto stream_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_error != cudaSuccess) {
            std::cerr << "Creating CUDA stream failed: " << cudaGetErrorString(stream_error) << '\n';
            return 1;
        }
        context._device_type = base::DeviceType::GPU;
        context._allocator = base::GPUAllocatorFactory::get_instance();
        context._stream = stream;
    }
    const int result = run_chat(options, context);
    if (stream != nullptr) {
        const auto error = cudaStreamDestroy(stream);
        if (error != cudaSuccess) {
            std::cerr << "Destroying CUDA stream failed: " << cudaGetErrorString(error) << '\n';
            return 1;
        }
    }
    google::ShutdownGoogleLogging();
    return result;
}
