#include "matmul_kernel.cuh"
#include <cstdint>
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
template <const int BLOCK_SIZE>
__global__ void matmul_kernel_cu_fp32(const float* input1, const float* input2, float scale,
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
            Bs[tx][ty] = input2[b_row * K + k0 + tx];
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

    constexpr int thread_size = 16;
    dim3 threads(thread_size, thread_size);
    dim3 blocks((M + thread_size - 1) / thread_size, (N + thread_size - 1) / thread_size);
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    if (_stream)
        matmul_kernel_cu_fp32<16><<<blocks, threads, 0, _stream>>>(
            input1.ptr<float>(), input2.ptr<float>(), scale, output.ptr<float>(), N, M, K);
    else {
        matmul_kernel_cu_fp32<16><<<blocks, threads>>>(input1.ptr<float>(), input2.ptr<float>(),
                                                       scale, output.ptr<float>(), N, M, K);
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
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    const float* in1 = input1.ptr<float>();
    const uint8_t* in2 = input2.ptr<uint8_t>();
    float* out = output.ptr<float>();
    const float* sca = scale.ptr<float>();
    const uint8_t* zero = zero_points.ptr<uint8_t>();

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    matmul_quant_int4_fp32_naive<<<grid, block, 0, _stream>>>(in1, in2, sca, zero, out, N, M, K,
                                                        group_size);
}

} // namespace kernel
