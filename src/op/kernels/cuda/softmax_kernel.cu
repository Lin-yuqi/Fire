#include "softmax_kernel.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace kernel {
namespace {

constexpr int block_size = 256;

__device__ __forceinline__ float blockreduce_mx(float val) {
    // warp内规约
    __shared__ float share[32];
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int lane = tid % 32;
    int warp_num = (blockDim.x + 31) / 32;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, tmp);
    }
    if (lane == 0) {
        share[warpid] = val;
    }
    __syncthreads();

    // warp级规约
    if (warpid == 0) {
        val = lane < warp_num ? share[lane] : -INFINITY;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val = fmaxf(val, tmp);
        }
    }
    return val;
}

__device__ __forceinline__ float blockreduce_sum(float val) {
    // warp内规约
    __shared__ float share[32];
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int lane = tid % 32;
    int warp_num = (blockDim.x + 31) / 32;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if (lane == 0) {
        share[warpid] = val;
    }
    __syncthreads();

    // warp级规约
    if (warpid == 0) {
        val = lane < warp_num ? share[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}

// C <= 2048 时，每个线程将 1/2/4/8 个元素保存在寄存器里。
template <int ITEMS_PER_THREAD>
__global__ void softmax_kernel_cu_fp32(int C, const float* input, float* output) {
    __shared__ float share;
    const int tid = threadIdx.x;
    const size_t row_offset = static_cast<size_t>(blockIdx.x) * C;
    const float* in = input + row_offset;
    float* out = output + row_offset;

    float vals[ITEMS_PER_THREAD];
    float mx = -INFINITY;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int col = tid + item * block_size;
        vals[item] = col < C ? in[col] : -INFINITY;
        mx = fmaxf(mx, vals[item]);
    }

    mx = blockreduce_mx(mx);
    if (tid == 0)
        share = mx;
    __syncthreads();
    mx = share;

    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        vals[item] = expf(vals[item] - mx);
        sum += vals[item];
    }

    sum = blockreduce_sum(sum);
    if (tid == 0)
        share = 1.0f / sum;
    __syncthreads();
    const float inv_sum = share;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        const int col = tid + item * block_size;
        if (col < C)
            out[col] = vals[item] * inv_sum;
    }
}

// 长行保留通用路径，避免为任意 C 分配过大的线程局部数组。
__global__ void softmax_kernel_cu_fp32_large(int C, const float* input, float* output) {
    __shared__ float _share;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    // 找到起始位置
    const size_t row_offset = static_cast<size_t>(C) * bid;
    const float* in = input + row_offset;
    float* out = output + row_offset;

    float mx = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        mx = fmaxf(mx, in[i]);
    }
    // 规约求最大值
    mx = blockreduce_mx(mx);
    if (tid == 0) {
        _share = mx;
    }
    __syncthreads();
    mx = _share;

    float sum = 0.f;
    for (int i = tid; i < C; i += blockDim.x) {
        const float value = expf(in[i] - mx);
        out[i] = value;
        sum += value;
    }

    sum = blockreduce_sum(sum);
    if (tid == 0) {
        _share = 1.0f / sum;
    }
    __syncthreads();
    const float inv_sum = _share;

    for (int i = tid; i < C; i += blockDim.x) {
        out[i] *= inv_sum;
    }
}

} // namespace

void softmax_kernel_cu(const tensor::Tensor& inuput, tensor::Tensor& output, void* stream) {
    int N = 1;
    int C = 1;

    if (inuput.dims_size() == 1) {
        C = inuput.get_dim(0);
    } else {
        N = inuput.get_dim(0);
        C = inuput.get_dim(1);
    }

    int block_num = N;

    const float* in = inuput.ptr<float>();
    float* out = output.ptr<float>();

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    if (C <= block_size) {
        softmax_kernel_cu_fp32<1><<<block_num, block_size, 0, _stream>>>(C, in, out);
    } else if (C <= block_size * 2) {
        softmax_kernel_cu_fp32<2><<<block_num, block_size, 0, _stream>>>(C, in, out);
    } else if (C <= block_size * 4) {
        softmax_kernel_cu_fp32<4><<<block_num, block_size, 0, _stream>>>(C, in, out);
    } else if (C <= block_size * 8) {
        softmax_kernel_cu_fp32<8><<<block_num, block_size, 0, _stream>>>(C, in, out);
    } else {
        softmax_kernel_cu_fp32_large<<<block_num, block_size, 0, _stream>>>(C, in, out);
    }
    CHECK(cudaGetLastError() == cudaSuccess);
}

} // namespace kernel
