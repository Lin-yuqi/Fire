#include "matmul_kernel.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
namespace kernel {
/*简单实现，一个线程搬运一个数据*/
// __global__ void matmul_kernel_cu_fp32(const float* input1, const float* input2, float scale,
//                                       float* output, const int N, const int M, const int K) {
//     int tx = blockDim.x*blockIdx.x+threadIdx.x;
//     int ty = blockDim.y*blockIdx.y+threadIdx.y;
//     if(tx>=N||ty>=M) return;
//     float sum = 0.f;
//     for(int i=0;i<K;i++){
//         sum += input1[tx*K+i]*input2[ty*K+i];
//     }
//     output[tx*M+ty]=sum*scale;
// }

// 通用矩阵乘v1共享显存,在运算时通过访问隐式转置
template <const int BLOCK_SIZE, typename WeightType>
__global__ void matmul_kernel_cu_dense(const float* input1, const WeightType* input2, float scale,
                                       float* output, const int N, const int M, const int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    const int n = blockIdx.y * BLOCK_SIZE + ty;
    const int m = blockIdx.x * BLOCK_SIZE + tx;
    float tmp = 0.f;
    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
        // -------------------------
        // load A tile
        //
        // A: [N, K]
        // -------------------------
        if (n < N && k0 + tx < K) {
            As[ty][tx] = input1[n * K + k0 + tx];
        } else {
            As[ty][tx] = 0.f;
        }
        // -------------------------
        // load B tile
        //
        // B: [M, K]
        //
        // 这里真正转置存入 shared
        // Bs: [K, M] 提高共享内存的访问效率
        // -------------------------
        const int b_row = blockIdx.x * BLOCK_SIZE + ty;

        if (b_row < M && k0 + tx < K) {
            // Widen BF16 at load; shared tiles and accumulation stay FP32.
            Bs[tx][ty] = static_cast<float>(input2[b_row * K + k0 + tx]);
        } else {
            Bs[tx][ty] = 0.f;
        }
        __syncthreads();

        // -------------------------
        // C[n,m]
        // =
        // A[n,:] · B[m,:]
        // -------------------------
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    if (n < N && m < M) {
        output[n * M + m] = tmp * scale;
    }
}

// One warp computes one output row; BF16 weights are widened before FP32 accumulation.
template <typename WeightType, bool Vectorized>
__global__ void GEMV_bf16_fp32_v1(const float* input, const WeightType* weight, float scale,
                                  float* output, int M, int K) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int m = blockIdx.x * (blockDim.x / warp_size) + threadIdx.x / warp_size;
    // The row index is uniform across the warp, including an inactive tail warp.
    if (m >= M) {
        return;
    }

    const WeightType* row = weight + static_cast<size_t>(m) * K;
    float4 sums = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (Vectorized) {
        for (int pack = lane; pack < K / 4; pack += warp_size) {
            const float4 values = reinterpret_cast<const float4*>(input)[pack];
            float4 weights;
            if constexpr (std::is_same_v<WeightType, float>) {
                weights = reinterpret_cast<const float4*>(row)[pack];
            } else {
                // One 64-bit load contains two pairs of BF16 weights.
                const uint2 packed = reinterpret_cast<const uint2*>(row)[pack];
                weights = make_float4(
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x))),
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.x >> 16))),
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y))),
                    __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(packed.y >> 16))));
            }
            // Independent accumulators shorten the dependency chain along K.
            sums.x += values.x * weights.x;
            sums.y += values.y * weights.y;
            sums.z += values.z * weights.z;
            sums.w += values.w * weights.w;
        }
    } else {
        // Handles arbitrary K and tensor views without vector-load alignment.
        for (int k = lane; k < K; k += warp_size) {
            sums.x += input[k] * static_cast<float>(row[k]);
        }
    }

    float sum = (sums.x + sums.y) + (sums.z + sums.w);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[m] = sum * scale;
    }
}

template <typename WeightType>
void launch_dense_gemv(const float* input, const WeightType* weight, float scale, float* output,
                       int M, int K, cudaStream_t stream) {
    constexpr int threads = 128;
    constexpr int rows_per_block = threads / 32;
    const int blocks = (M + rows_per_block - 1) / rows_per_block;
    // K%4 also keeps every weight row aligned for its vector type.
    if (K % 4 == 0 && reinterpret_cast<uintptr_t>(input) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(weight) % (4 * sizeof(WeightType)) == 0) {
        GEMV_bf16_fp32_v1<WeightType, true><<<blocks, threads, 0, stream>>>(
            input, weight, scale, output, M, K);
    } else {
        GEMV_bf16_fp32_v1<WeightType, false><<<blocks, threads, 0, stream>>>(
            input, weight, scale, output, M, K);
    }
}

// input1可以是1D/2D,input2是2D
void matmul_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2, float scale,
                      tensor::Tensor& output, void* stream) {
    CHECK(input1.is_empty() == false && input1.dims_size() <= 2);
    CHECK(input1.device_type() == base::DeviceType::GPU);

    CHECK(input2.is_empty() == false && input2.dims_size() == 2);
    CHECK(input2.device_type() == base::DeviceType::GPU);
    /*
    input1:
        1D: [K]
        2D: [N, K]

    input2:
        2D: [M, K]

    output:
        input1 是 1D -> [M]
        input1 是 2D -> [N, M]
    */
    int N = 1;
    int K = 0;
    if (input1.dims_size() == 1) {
        K = input1.get_dim(0);
    } else {
        N = input1.get_dim(0);
        K = input1.get_dim(1);
    }

    const int M = input2.get_dim(0);
    CHECK_EQ(input2.get_dim(1), K);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    if (N == 1) {
        if (input2.data_type() == base::DataType::Bf16) {
            launch_dense_gemv(input1.ptr<float>(), input2.ptr<__nv_bfloat16>(), scale,
                              output.ptr<float>(), M, K, _stream);
        } else {
            launch_dense_gemv(input1.ptr<float>(), input2.ptr<float>(), scale,
                              output.ptr<float>(), M, K, _stream);
        }
        return;
    }

    constexpr int thread_size = 16;
    dim3 threads(thread_size, thread_size);
    dim3 blocks((M + thread_size - 1) / thread_size, (N + thread_size - 1) / thread_size);
    if (input2.data_type() == base::DataType::Bf16) {
        matmul_kernel_cu_dense<thread_size><<<blocks, threads, 0, _stream>>>(
            input1.ptr<float>(), input2.ptr<__nv_bfloat16>(), scale, output.ptr<float>(), N, M, K);
    } else {
        matmul_kernel_cu_dense<thread_size><<<blocks, threads, 0, _stream>>>(
            input1.ptr<float>(), input2.ptr<float>(), scale, output.ptr<float>(), N, M, K);
    }
}

// 不就是matmul吗？先实现简单的，一个元素一个线程
__global__ void matmul_quant_int4_fp32_naive(const float* input, const uint8_t* qweight,
                                             const float* scales, const uint8_t* zero_points,
                                             float* output, int N, int M, int K, int group_size) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
        return;

    const int pack_k = K / 2;
    const int group_per_row = K / group_size;
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        const uint8_t packd = qweight[m * pack_k + k / 2];
        const int q = (k & 1) ? (packd >> 4) : (packd & 0x0f);

        const int group = k / group_size;
        const int metadata_idx = m * group_per_row + group;
        const float weight =
            (q - static_cast<int>(zero_points[metadata_idx])) * scales[metadata_idx];

        sum += input[n * K + k] * weight;
    }

    output[n * M + m] = sum;
}

// group_size=128: one warp computes one row, four weights per lane and group.
__global__ void GEMV_int4_v1(const float* input, const uint8_t* qweight, const float* scales,
                            const uint8_t* zero_points, float* output, int M, int K) {
    constexpr int warp_size = 32;
    constexpr int group_size = 128;
    const int lane = threadIdx.x % warp_size;
    const int m = blockIdx.x * (blockDim.x / warp_size) + threadIdx.x / warp_size;
    // All lanes in an out-of-range warp leave before the warp reduction.
    if (m >= M) {
        return;
    }

    const int groups_per_row = K / group_size;
    const auto* packed_row = reinterpret_cast<const uint16_t*>(
        qweight + static_cast<size_t>(m) * (K / 2));
    const auto* input_packs = reinterpret_cast<const float4*>(input);
    float sum = 0.0f;
    for (int group = 0; group < groups_per_row; ++group) {
        const size_t metadata_idx = static_cast<size_t>(m) * groups_per_row + group;
        const float scale = scales[metadata_idx];
        const int zero = zero_points[metadata_idx];
        const int pack = group * (group_size / 4) + lane;

        // Two packed bytes contain two adjacent pairs of INT4 weights.
        const uint16_t weights = packed_row[pack];
        const float4 values = input_packs[pack];
        sum += values.x * ((static_cast<int>(weights & 0x0f) - zero) * scale);
        sum += values.y * ((static_cast<int>((weights >> 4) & 0x0f) - zero) * scale);
        sum += values.z * ((static_cast<int>((weights >> 8) & 0x0f) - zero) * scale);
        sum += values.w * ((static_cast<int>((weights >> 12) & 0x0f) - zero) * scale);
    }

    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[m] = sum;
    }
}
/*
input1:
    1D: [K]
    2D: [N, K]

input2:
    2D: [M, K]

output:
    input1 是 1D -> [M]
    input1 是 2D -> [N, M]
*/
void matmul_quant_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            tensor::Tensor& output, int32_t group_size, const tensor::Tensor& scale,
                            const tensor::Tensor& zero_points, void* stream) {
    CHECK(input1.is_empty() == false && input1.dims_size() <= 2);
    CHECK(input1.device_type() == base::DeviceType::GPU);

    CHECK(input2.is_empty() == false && input2.dims_size() == 2);
    CHECK(input2.device_type() == base::DeviceType::GPU);

    const int N = input1.dims_size() == 1 ? 1 : input1.get_dim(0);
    const int K = input1.get_dim(input1.dims_size() - 1);
    const int M = input2.get_dim(0);
    CHECK_EQ(input2.get_dim(1), K / 2);
    const float* in1 = input1.ptr<float>();
    const uint8_t* in2 = input2.ptr<uint8_t>();
    float* out = output.ptr<float>();
    const float* sca = scale.ptr<float>();
    const uint8_t* zero = zero_points.ptr<uint8_t>();

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    if (N == 1 && group_size == 128 && K % 128 == 0 &&
        reinterpret_cast<uintptr_t>(in1) % alignof(float4) == 0 &&
        reinterpret_cast<uintptr_t>(in2) % alignof(uint16_t) == 0) {
        constexpr int threads = 128;
        constexpr int rows_per_block = threads / 32;
        GEMV_int4_v1<<<(M + rows_per_block - 1) / rows_per_block, threads, 0, _stream>>>(
            in1, in2, sca, zero, out, M, K);
        return;
    }

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_quant_int4_fp32_naive<<<grid, block, 0, _stream>>>(in1, in2, sca, zero, out, N, M, K,
                                                              group_size);
}

} // namespace kernel
