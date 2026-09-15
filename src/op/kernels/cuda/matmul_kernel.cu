#include "matmul_kernel.cuh"
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

    ;
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

} // namespace kernel
