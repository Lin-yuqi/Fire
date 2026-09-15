#include "emb_kernel.cuh"

namespace kernel {

// 每个 block 负责一个 token，对应搬运一行 embedding
__global__ void emb_kernel_cu_fp32(int32_t dim, const int32_t* input, const float* weight,
                                   float* output) {
    constexpr int pack_size = 4;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    // 当前输入位置对应的 token id
    const int token_id = input[bid];
    // weight 中这一行的偏移
    const int weight_offset = token_id * dim;
    // output 中这一行的偏移
    const int output_offset = bid * dim;
    /*
     * cudaMalloc 得到的基地址本身是足够对齐的，
     * 所以这里只需要判断行偏移是否为 4 个 float 的倍数。
     *
     * weight 行和 output 行都必须对齐，才能安全使用 float4。
     */
    const bool row_is_aligned =
        (weight_offset % pack_size == 0) && (output_offset % pack_size == 0);

    // 不对齐：pack_num = 0，整行走 scalar
    const int pack_num = row_is_aligned ? dim / pack_size : 0;

    const int pack_off = pack_num * pack_size;

    const float* weight_row = weight + weight_offset;
    float* output_row = output + output_offset;

    // 只有 row_is_aligned 时才真正访问这些 float4
    const float4* weight_pack = reinterpret_cast<const float4*>(weight_row);

    float4* output_pack = reinterpret_cast<float4*>(output_row);

    // -------- float4 路径 --------
    for (int cur = tid; cur < pack_num; cur += blockDim.x) {

        output_pack[cur] = weight_pack[cur];
    }

    // -------- scalar 路径 / 尾部 --------
    for (int cur = pack_off + tid; cur < dim; cur += blockDim.x) {

        output_row[cur] = weight_row[cur];
    }
}

void emb_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                   tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    int32_t input_num = input.size();
    int32_t dim = weight.get_dim(1);

    constexpr int thread_num = 256;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    const int32_t* in = input.ptr<int32_t>();
    const float* wei = weight.ptr<float>();
    float* out = output.ptr<float>();
    if (_stream)
        emb_kernel_cu_fp32<<<input_num, thread_num, 0, _stream>>>(dim, in, wei, out);
    else
        emb_kernel_cu_fp32<<<input_num, thread_num>>>(dim, in, wei, out);
}

} // namespace kernel
