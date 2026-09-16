#include "rope_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kernel {

template <int POS_PER_BLOCK>
__global__ void sin_cos_cache_kernel_cu_fp32(int head_size, int max_seq_len, float rope_theta,
                                             float* sin_cache, float* cos_cache) {
    const int half_size = head_size / 2;
    const int pos_begin = blockIdx.x * POS_PER_BLOCK;

    for (int d = threadIdx.x; d < half_size; d += blockDim.x) {
        // 一个线程在多个位置间复用频率，cache 只保存每对维度的一份 sin/cos。
        const float freq =
            1.0f / powf(rope_theta, 2.0f * static_cast<float>(d) / head_size);
#pragma unroll
        for (int p = 0; p < POS_PER_BLOCK; ++p) {
            const int pos = pos_begin + p;
            if (pos >= max_seq_len)
                break;

            float sin_value, cos_value;
            sincosf(static_cast<float>(pos) * freq, &sin_value, &cos_value);
            const size_t offset = static_cast<size_t>(pos) * half_size + d;
            sin_cache[offset] = sin_value;
            cos_cache[offset] = cos_value;
        }
    }
}

// 一个 block 处理 8 个位置，线程沿 cache 的连续维度分布。
void sin_cos_cache_kernel_cu(int head_size, int max_seq_len, float rope_theta,
                             tensor::Tensor& sin_cache, tensor::Tensor& cos_cache, void* stream) {
    constexpr int positions_per_block = 8;
    const int thread_num = std::min(head_size / 2, 256);
    const int block_num = (max_seq_len + positions_per_block - 1) / positions_per_block;

    float* sin = sin_cache.ptr<float>();
    float* cos = cos_cache.ptr<float>();
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    sin_cos_cache_kernel_cu_fp32<positions_per_block><<<block_num, thread_num, 0, _stream>>>(
        head_size, max_seq_len, rope_theta, sin, cos);
    CHECK(cudaGetLastError() == cudaSuccess);
}

// 一个线程处理一个 head 内前后半区的一对元素。
__global__ void rope_kernel_cu_fp32(float* q, float* k, const float* cos, const float* sin,
                                    int32_t dim, int32_t kv_dim, int32_t head_size) {

    const int32_t pair_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (pair_idx >= dim / 2)
        return;

    const int32_t half_size = head_size / 2;
    const int32_t head_dim = pair_idx % half_size;
    const int32_t idx = (pair_idx / half_size) * head_size + head_dim;

    // 取数据
    float fci = sin[head_dim];
    float fcr = cos[head_dim];
    // 算q
    float v0 = q[idx];
    float v1 = q[idx + half_size];
    q[idx] = v0 * fcr - v1 * fci;
    q[idx + half_size] = v0 * fci + v1 * fcr;
    // 算k
    if (idx < kv_dim) {
        v0 = k[idx];
        v1 = k[idx + half_size];
        k[idx] = v0 * fcr - v1 * fci;
        k[idx + half_size] = v0 * fci + v1 * fcr;
    }
}

void rope_kernel_cu(tensor::Tensor& input_q, tensor::Tensor& input_k, const tensor::Tensor& cos,
                    const tensor::Tensor& sin, int32_t pos, void* stream) {
    int32_t head_size = input_q.get_dim(1);
    int32_t dim = input_q.get_dim(0) * input_q.get_dim(1);
    int32_t kv_dim = input_k.get_dim(0) * input_k.get_dim(1);

    int32_t thread_num = 256;

    int32_t block_num = (dim / 2 + thread_num - 1) / thread_num;

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    float* q = input_q.ptr<float>();
    float* k = input_k.ptr<float>();

    const size_t offset = static_cast<size_t>(pos) * (head_size / 2);
    const float* _cos = cos.ptr<float>() + offset;
    const float* _sin = sin.ptr<float>() + offset;

    rope_kernel_cu_fp32<<<block_num, thread_num, 0, _stream>>>(q, k, _cos, _sin, dim, kv_dim,
                                                               head_size);

    CHECK(cudaGetLastError() == cudaSuccess);
}
} // namespace kernel
