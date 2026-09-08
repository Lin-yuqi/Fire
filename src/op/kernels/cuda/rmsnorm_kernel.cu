#include "rmsnorm_kernel.cuh"
#include <cub/block/block_reduce.cuh>

// namespace {
// //--------------------- for practice -------------
// __global__ void block_reduce(float* in, float* out, int sz) {
//     __shared__ float share[32];
//     int tid = threadIdx.x;
//     float sum = 0;
//     for (int i = tid; i < sz; i += blockDim.x) {
//         sum += in[i];
//     }

//     int laneid = tid % 32;
//     int warpid = tid / 32;

//     for (int i = 16; i; i >>= 1) {
//         sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
//     }
//     __syncthreads();
//     if (laneid == 0) {
//         share[warpid] = sum;
//     }
//     __syncthreads();
//     if (warpid == 0) {
//         sum = tid < (blockDim.x + 31 / 32) ? share[tid] : 0;
//         for (int i = 16; i; i >>= 1) {
//             sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
//         }
//     }
//     if (tid == 0) {
//         share[0] = sum;
//     }
//     __syncthreads();
//     sum = share[0];
// }
// } // namespace

namespace kernel {

//dim版本和非dim版本的区别就是是否是一次性处理多行数据

__global__ void rmsnorm_kernel_cu_fp32_dim(float* in, float* wei, float* out, const int size,
                                           const float eps) {
    const int tid = threadIdx.x;
    constexpr int pack_size = 4;
    const int offset = size * blockIdx.x;
    const bool row_is_aligned = offset % pack_size == 0;
    // 不对齐时pack_num = 0 直接走标量路径
    const int pack_num = row_is_aligned ? size / pack_size : 0;
    const int pack_off = pack_size * pack_num;

    float sum = 0.0f;
    float4* inpack = reinterpret_cast<float4*>(in + offset);
    float4* outpack = reinterpret_cast<float4*>(out + offset);
    float4* weipack = reinterpret_cast<float4*>(wei);
    //这里weight不跟着行偏移


    float4 vals[16];
    for (int i = tid, k = 0; i < pack_num; i += blockDim.x, k++) {
        float4 in_float4 = inpack[i];
        vals[k] = in_float4;

        sum += vals[k].x * vals[k].x;
        sum += vals[k].y * vals[k].y;
        sum += vals[k].z * vals[k].z;
        sum += vals[k].w * vals[k].w;
    }

    float* row_in = in + offset;
    float* row_out = out + offset;

    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += row_in[i] * row_in[i];
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float share_val;

    sum = BlockReduce(temp).Sum(sum);

    if (tid == 0) {
        share_val = sum;
    }
    __syncthreads();

    sum = share_val;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int i = tid, k = 0; i < pack_num; i += blockDim.x, k++) {
        float4 w = weipack[i];
        float4 in = vals[k];
        outpack[i] = make_float4(scale * w.x * in.x, scale * w.y * in.y, scale * w.z * in.z,
                                 scale * w.w * in.w);
    }
    for (int i = tid + pack_off; i < size; i += blockDim.x) {
        row_out[i] = scale * wei[i] * row_in[i];
    }
}

template <int32_t BLOCK_DIM>
__global__ void rmsnorm_kernel_cu_fp32(float* in, float* wei, float* out, const int size,
                                       const float eps) {
    const int tid = threadIdx.x;
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    const int pack_off = pack_size * pack_num;

    float sum = 0.0f;
    float4* inpack = reinterpret_cast<float4*>(in);
    float4* outpack = reinterpret_cast<float4*>(out);
    float4* weipack = reinterpret_cast<float4*>(wei);
    float4 vals[16];
    for (int i = tid, k = 0; i < pack_num; i += blockDim.x, k++) {
        float4 in_float4 = inpack[i];
        vals[k] = in_float4;

        sum += vals[k].x * vals[k].x;
        sum += vals[k].y * vals[k].y;
        sum += vals[k].z * vals[k].z;
        sum += vals[k].w * vals[k].w;
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float share_val;

    sum = BlockReduce(temp).Sum(sum);

    if (tid == 0) {
        share_val = sum;
    }
    __syncthreads();

    sum = share_val;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int i = tid, k = 0; i < pack_num; i += blockDim.x, k++) {
        float4 w = weipack[i];
        float4 in = vals[k];
        outpack[i] = make_float4(scale * w.x * in.x, scale * w.y * in.y, scale * w.z * in.z,
                                 scale * w.w * in.w);
    }
    for (int i = tid + pack_off; i < size; i += blockDim.x) {
        out[i] = scale * wei[i] * in[i];
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       tensor::Tensor& output, const float eps, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::GPU &&
          weight.device_type() == base::DeviceType::GPU &&
          output.device_type() == base::DeviceType::GPU);

    float* in = const_cast<float*>(input.ptr<float>());
    float* wei = const_cast<float*>(weight.ptr<float>());
    float* out = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 256;
    int32_t size = input.size();
    if (stream) {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        rmsnorm_kernel_cu_fp32<256><<<1, threads_num, 0, _stream>>>(in, wei, out, size, eps);
    } else {
        rmsnorm_kernel_cu_fp32<256><<<1, threads_num>>>(in, wei, out, size, eps);
    }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           tensor::Tensor& output, const float eps, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::GPU &&
          weight.device_type() == base::DeviceType::GPU &&
          output.device_type() == base::DeviceType::GPU);

    float* in = const_cast<float*>(input.ptr<float>());
    float* wei = const_cast<float*>(weight.ptr<float>());
    float* out = const_cast<float*>(output.ptr<float>());

    constexpr int threads_num = 256;
    int32_t size = input.dims().back();
    int blocks_num = input.dims()[0];
    if (stream) {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        rmsnorm_kernel_cu_fp32_dim<<<blocks_num, threads_num, 0, _stream>>>(in, wei, out, size,
                                                                            eps);
    } else {
        rmsnorm_kernel_cu_fp32_dim<<<blocks_num, threads_num>>>(in, wei, out, size, eps);
    }
}

} // namespace kernel
