#include "Fire/sampler/argmax_sampler.h"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

struct ArgmaxCase {
    std::vector<float> logits;
    size_t expected_index;
};

std::vector<ArgmaxCase> make_cases() {
    std::vector<float> vocabulary_logits(32000, -1000.0f);
    vocabulary_logits[17] = 8.0f;
    vocabulary_logits[30001] = 8.0f;

    return {
        {{42.0f}, 0},
        {{-9.0f, -1.0f, -4.0f}, 1},
        {{1.0f, 5.0f, 5.0f, 2.0f}, 1},
        {std::move(vocabulary_logits), 17},
    };
}

void expect_gpu_argmax(const ArgmaxCase& test_case, cudaStream_t stream) {
    SCOPED_TRACE(::testing::Message() << "size=" << test_case.logits.size());

    float* device_logits = nullptr;
    const size_t byte_size = test_case.logits.size() * sizeof(float);
    ASSERT_EQ(cudaMalloc(&device_logits, byte_size), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(device_logits, test_case.logits.data(), byte_size,
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);

    sampler::ArgmaxSampler sampler(base::DeviceType::GPU);
    EXPECT_EQ(sampler.sample(device_logits, test_case.logits.size(), stream),
              test_case.expected_index);

    EXPECT_EQ(cudaFree(device_logits), cudaSuccess);
}

TEST(ArgmaxSamplerTest, CpuReturnsFirstMaximumIndex) {
    sampler::ArgmaxSampler sampler(base::DeviceType::CPU);

    for (const auto& test_case : make_cases()) {
        SCOPED_TRACE(::testing::Message() << "size=" << test_case.logits.size());
        EXPECT_EQ(sampler.sample(test_case.logits.data(), test_case.logits.size()),
                  test_case.expected_index);
    }
}

class ArgmaxSamplerCudaTest : public ::testing::Test {
  protected:
    void SetUp() override {
        int device_count = 0;
        const cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
            (error == cudaSuccess && device_count == 0)) {
            GTEST_SKIP() << "CUDA unavailable";
        }
        ASSERT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override {
        if (stream != nullptr) {
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    }

    cudaStream_t stream = nullptr;
};

TEST_F(ArgmaxSamplerCudaTest, MatchesCpuOnDefaultStream) {
    for (const auto& test_case : make_cases()) {
        ASSERT_NO_FATAL_FAILURE(expect_gpu_argmax(test_case, nullptr));
    }
}

TEST_F(ArgmaxSamplerCudaTest, MatchesCpuOnNonDefaultStream) {
    for (const auto& test_case : make_cases()) {
        ASSERT_NO_FATAL_FAILURE(expect_gpu_argmax(test_case, stream));
    }
}

} // namespace
