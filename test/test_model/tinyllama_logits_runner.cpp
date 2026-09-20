#include "Fire/base/alloc.h"
#include "Fire/model/tinyllama.h"
#include "Fire/model/tinyllama_loader.h"

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
    std::cerr << "tinyllama_logits_runner: " << message << '\n';
    return 1;
}

bool parse_token_id(const char* text, int32_t& token_id) {
    const std::string value(text);
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), token_id);
    return error == std::errc{} && end == value.data() + value.size() && token_id >= 0 &&
           token_id < model::TinyLlamaProfile::model.vocab_size;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        return fail("usage: tinyllama_logits_runner <model.fire> <output.bin> <token-id>...");
    }

    std::vector<int32_t> token_ids;
    token_ids.reserve(static_cast<size_t>(argc - 3));
    for (int argument = 3; argument < argc; ++argument) {
        int32_t token_id = 0;
        if (!parse_token_id(argv[argument], token_id)) {
            return fail("invalid token id: " + std::string(argv[argument]));
        }
        token_ids.push_back(token_id);
    }

    model::TinyllamaLoader loader;
    auto status = loader.open(argv[1]);
    if (!status) {
        return fail("cannot open Fire model: " + status.message());
    }

    model::TinyLlamaWeights weights;
    status = loader.load_weights(weights);
    if (!status) {
        return fail("cannot load Fire weights: " + status.message());
    }

    op::OpContext context;
    context._device_type = base::DeviceType::CPU;
    context._allocator = base::CPUAllocatorFactory::get_instance();

    std::unique_ptr<model::TinyLlamaModel> tinyllama;
    status = model::TinyLlamaModel::create(weights, context, tinyllama);
    if (!status) {
        return fail("cannot create Fire model: " + status.message());
    }

    status = tinyllama->prepare(static_cast<int32_t>(token_ids.size()), context);
    if (!status) {
        return fail("cannot prepare Fire model: " + status.message());
    }

    const size_t vocab_size = static_cast<size_t>(tinyllama->config().vocab_size);
    std::vector<float> all_logits(token_ids.size() * vocab_size);
    tensor::Tensor logits(base::DataType::Fp32, {tinyllama->config().vocab_size},
                          context._allocator);

    for (size_t position = 0; position < token_ids.size(); ++position) {
        status = tinyllama->forward(token_ids[position], static_cast<int32_t>(position), logits,
                                    context);
        if (!status) {
            return fail("forward failed at position " + std::to_string(position) + ": " +
                        status.message());
        }
        std::copy_n(logits.ptr<float>(), vocab_size,
                    all_logits.data() + position * vocab_size);
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
