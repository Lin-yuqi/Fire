#include "argmax_kernel.cuh"

#include "Fire/base/alloc.h"

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>
#include <cuda_runtime.h>

#include <cmath>

namespace kernel {
namespace {

constexpr int kBlockSize = 256;
using ArgMaxPair = cub::KeyValuePair<size_t, float>;

__global__ void argmax_kernel_fp32(const float* input, size_t size, size_t* output) {
    // key 是下标，value 是 logit。
    // cub::ArgMax 在数值相同时选择较小的下标，与 std::max_element 一致。
    ArgMaxPair local_max(0, -INFINITY);

    for (size_t index = threadIdx.x; index < size; index += blockDim.x) {
        local_max = cub::ArgMax{}(local_max, ArgMaxPair(index, input[index]));
    }

    using BlockReduce = cub::BlockReduce<ArgMaxPair, kBlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const ArgMaxPair block_max = BlockReduce(temp_storage).Reduce(local_max, cub::ArgMax{});

    if (threadIdx.x == 0) {
        *output = block_max.key;
    }
}

} // namespace

size_t argmax_kernel_cu(const float* input, size_t size, void* stream) {
    CHECK(input != nullptr);
    CHECK_GT(size, 0U);

    const auto allocator = base::GPUAllocatorFactory::get_instance();
    auto* device_index = static_cast<size_t*>(allocator->allocate(sizeof(size_t)));
    CHECK(device_index != nullptr);

    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    argmax_kernel_fp32<<<1, kBlockSize, 0, cuda_stream>>>(input, size, device_index);
    CHECK(cudaGetLastError() == cudaSuccess);

    size_t host_index = 0;
    CHECK(cudaMemcpyAsync(&host_index, device_index, sizeof(size_t), cudaMemcpyDeviceToHost,
                          cuda_stream) == cudaSuccess);

    // 当前接口直接返回 CPU 标量，因此这里必须同步。
    CHECK(cudaStreamSynchronize(cuda_stream) == cudaSuccess);

    allocator->release(device_index);
    return host_index;
}

} // namespace kernel
