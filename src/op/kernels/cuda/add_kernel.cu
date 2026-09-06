#include "add_kernel.cuh"


// -----------------kernel begin------------------
namespace kernel {

__global__ void add_kernel_cu_fp32(int32_t size, const float* in1, const float* in2, float* out) {
    int32_t tid=blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>=size)return ;
     out[tid] = in1[tid] + in2[tid];
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   tensor::Tensor& output, void* stream) {
    CHECK(!input1.is_empty());
    CHECK(!input2.is_empty());
    CHECK(!output.is_empty());

    int32_t size = input1.size();
    CHECK_EQ(size, input2.size());
    CHECK_EQ(size, output.size());
    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;
    if (stream) {
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        add_kernel_cu_fp32<<<block_num, thread_num, 0, _stream>>>(
            size, input1.ptr<float>(), input2.ptr<float>(), output.ptr<float>());
    } else {
        add_kernel_cu_fp32<<<block_num, thread_num>>>(size, input1.ptr<float>(),
                                                      input2.ptr<float>(), output.ptr<float>());
    }
}

} // namespace kernel
// -----------------kernel end--------------------