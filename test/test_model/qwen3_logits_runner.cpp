#include "Fire/base/alloc.h"
#include "Fire/model/qwen.h"
#include "Fire/model/qwen_loader.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "qwen3_logits_runner: " << message << '\n';
    return 1;
}

bool parse_token_id(const char* text, int32_t vocab_size, int32_t& token_id) {
    const std::string value(text);
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), token_id);
    return error == std::errc{} && end == value.data() + value.size() && token_id >= 0 &&
           token_id < vocab_size;
}

} // namespace

int main(int argc, char** argv) {
    const std::string usage =
        "usage: qwen3_logits_runner <model.fire> <output.bin> [--gpu] "
        "[--profile 0.6b|8b] <token-id>...";
    if (argc < 4) {
        return fail(usage);
    }

    bool use_gpu = false;
    const model::Qwen3Profile* selected_profile = &model::qwen3_profiles::Qwen3_0_6B;
    int first_token_argument = 3;
    while (first_token_argument < argc) {
        const std::string argument(argv[first_token_argument]);
        if (argument == "--gpu") {
            use_gpu = true;
            ++first_token_argument;
        } else if (argument == "--profile") {
            if (first_token_argument + 1 >= argc) {
                return fail("--profile requires 0.6b or 8b");
            }
            const std::string name(argv[first_token_argument + 1]);
            if (name == "0.6b") {
                selected_profile = &model::qwen3_profiles::Qwen3_0_6B;
            } else if (name == "8b") {
                selected_profile = &model::qwen3_profiles::Qwen3_8B;
            } else {
                return fail("unsupported profile: " + name);
            }
            first_token_argument += 2;
        } else {
            break;
        }
    }
    if (argc <= first_token_argument) {
        return fail(usage);
    }

    const auto& profile = *selected_profile;
    std::vector<int32_t> token_ids;
    token_ids.reserve(static_cast<size_t>(argc - first_token_argument));
    for (int argument = first_token_argument; argument < argc; ++argument) {
        int32_t token_id = 0;
        if (!parse_token_id(argv[argument], profile.model.vocab_size, token_id)) {
            return fail("invalid token id: " + std::string(argv[argument]));
        }
        token_ids.push_back(token_id);
    }

    model::Qwen3Loader loader(profile);
    auto status = loader.open(argv[1]);
    if (!status) {
        return fail("cannot open Fire model: " + status.message());
    }

    model::Qwen3Weights weights;
    status = loader.load_weights(weights);
    if (!status) {
        return fail("cannot load Fire weights: " + status.message());
    }

    op::OpContext context;
    if (use_gpu) {
        int device_count = 0;
        const auto cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess) {
            return fail("CUDA device check failed: " + std::string(cudaGetErrorString(cuda_status)));
        }
        if (device_count == 0) {
            return fail("no CUDA device is available");
        }
        context._device_type = base::DeviceType::GPU;
        context._allocator = base::GPUAllocatorFactory::get_instance();
    } else {
        context._device_type = base::DeviceType::CPU;
        context._allocator = base::CPUAllocatorFactory::get_instance();
    }

    std::unique_ptr<model::Qwen3Model> qwen3;
    status = model::Qwen3Model::create(weights, context, qwen3);
    if (!status) {
        return fail("cannot create Fire model: " + status.message());
    }

    status = qwen3->prepare(static_cast<int32_t>(token_ids.size()), context);
    if (!status) {
        return fail("cannot prepare Fire model: " + status.message());
    }

    const size_t vocab_size = static_cast<size_t>(qwen3->config().vocab_size);
    std::vector<float> all_logits(token_ids.size() * vocab_size);
    tensor::Tensor logits(base::DataType::Fp32, {qwen3->config().vocab_size}, context._allocator);

    for (size_t position = 0; position < token_ids.size(); ++position) {
        status = qwen3->forward(token_ids[position], static_cast<int32_t>(position), logits,
                                context);
        if (!status) {
            return fail("forward failed at position " + std::to_string(position) + ": " +
                        status.message());
        }
        float* destination = all_logits.data() + position * vocab_size;
        if (use_gpu) {
            const auto cuda_status = cudaMemcpy(destination, logits.ptr<float>(),
                                                logits.byte_size(), cudaMemcpyDeviceToHost);
            if (cuda_status != cudaSuccess) {
                return fail("cannot copy GPU logits at position " + std::to_string(position) +
                            ": " + cudaGetErrorString(cuda_status));
            }
        } else {
            std::copy_n(logits.ptr<float>(), vocab_size, destination);
        }
    }

    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) {
        return fail("cannot open output file: " + std::string(argv[2]));
    }
    output.write(reinterpret_cast<const char*>(all_logits.data()),
                 static_cast<std::streamsize>(all_logits.size() * sizeof(float)));
    if (!output) {
        return fail("cannot write output file: " + std::string(argv[2]));
    }
    return 0;
}
