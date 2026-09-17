#include "mha_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <cfloat>
#include <cmath>

namespace kernel {

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    // this should be FLT_MAX, not 0 !!!!
    // otherwise, the softmax may be occur nan when head_dim < 128 threads
    float max_val = tid < size ? x[tid] : -FLT_MAX;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cuda::maximum<>{});
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

// one block one head
__global__ void mha_kernel_cu_fp32(int32_t num_q_heads, int32_t num_kv_heads, int32_t head_size,
                                   int32_t capacity, int32_t kv_mul, int32_t pos,
                                   int32_t layer_offset, const float* query,
                                   const float* key_cache, const float* val_cache, float* score,
                                   float* output) {
    int tid = threadIdx.x;
    int head_idx = blockIdx.x;
    const float* q_head = query + head_idx * head_size;
    extern __shared__ float s_query_head[];
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        s_query_head[i] = q_head[i];
    }

    int head_offset = head_idx / kv_mul * head_size;

    float scale = 1.0f / sqrtf(static_cast<float>(head_size));

    float* score_head = score + head_idx * capacity;

    __syncthreads();
    // Q head * K head * scale
    for (int t = tid; t <= pos; t += blockDim.x) { // 一个头一个head 和多个token相乘
        const float* key_head =
            key_cache + layer_offset + head_offset + t * num_kv_heads * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; ++i) {
            score += key_head[i] * s_query_head[i];
        }
        score_head[t] = score * scale;
    }
    __syncthreads();
    // softmax
    softmax_gpu(score_head, pos + 1);
    __syncthreads();

    // score * V
    float* output_head = output + head_idx * head_size;

    for (int i = tid; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val_head =
                val_cache + layer_offset + head_offset + t * num_kv_heads * head_size;
            const float attention_weight = score_head[t];
            val += attention_weight * val_head[i];
        }
        output_head[i] = val;
    }
}
// q:         [num_attention_heads, head_dim]
// key_cache: [num_layers, capacity, num_kv_heads, head_dim]
// val_cache: [num_layers, capacity, num_kv_heads, head_dim]
// score:     [num_attention_heads, capacity]
// output:    [num_attention_heads, head_dim]
void mha_kernel_cu(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                   const tensor::Tensor& val_cache, tensor::Tensor& score, tensor::Tensor& output,
                   int32_t layer_idx, int32_t pos, void* stream) {
    int32_t num_q_heads = input_q.get_dim(0);
    int32_t head_dim = input_q.get_dim(1);

    int32_t num_layers = key_cache.get_dim(0);
    int32_t capacity = key_cache.get_dim(1);
    int32_t num_kv_heads = key_cache.get_dim(2);
    int32_t kv_head_dim = key_cache.get_dim(3);

    CHECK(layer_idx >= 0 && layer_idx < num_layers);
    CHECK(pos >= 0 && pos < capacity);

    CHECK(head_dim == kv_head_dim);
    CHECK(num_q_heads % num_kv_heads == 0);

    CHECK(score.get_dim(0) == num_q_heads);
    CHECK(score.get_dim(1) == capacity);

    CHECK(output.get_dim(0) == num_q_heads);
    CHECK(output.get_dim(1) == head_dim);

    int32_t kv_mul = num_q_heads / num_kv_heads;

    int32_t layer_offset = layer_idx * capacity * num_kv_heads * head_dim;

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    const float* query = input_q.ptr<float>();
    const float* k_cache = key_cache.ptr<float>();
    const float* v_cache = val_cache.ptr<float>();
    float* score_cache = score.ptr<float>();
    float* out = output.ptr<float>();

    int32_t thread_num = 256;
    int32_t block_num = num_q_heads;

    mha_kernel_cu_fp32<<<block_num, thread_num, head_dim * sizeof(float), _stream>>>(
        num_q_heads, num_kv_heads, head_dim, capacity, kv_mul, pos, layer_offset, query, k_cache,
        v_cache, score_cache, out);
}

} // namespace kernel
