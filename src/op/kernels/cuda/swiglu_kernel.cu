#include "swiglu_kernel.cuh"
namespace kernel {

__global__ void swiglu_kernel_cu_fp32(int size, const float* in1, const float* in2, float* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= size)
        return;

    float x = in1[idx];
    float silu = x * 1.0f / (1.0f + expf(-x));
    out[idx] = silu * in2[idx];
}

void swiglu_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                      tensor::Tensor& output, void* stream) {
    int sz = input1.size();
    int thread_num = 256;
    int block_num = (sz + thread_num - 1) / thread_num;

    const float* in1 = input1.ptr<float>();
    const float* in2 = input2.ptr<float>();

    float* out = output.ptr<float>();

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    swiglu_kernel_cu_fp32<<<block_num, thread_num, 0, _stream>>>(sz, in1, in2, out);
}

} // namespace kernel