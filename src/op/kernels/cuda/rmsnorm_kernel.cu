#include "rmsnorm_kernel.cuh"
#include <__clang_cuda_builtin_vars.h>



namespace  {
//--------------------- for practice -------------
__global__ void block_reduce(float* in,float* out,int sz){
    __shared__ float share[32];
    int tid=threadIdx.x;
    float sum=0;
    for(int i=tid;i<sz;i+=blockDim.x){
        sum+=in[i];
    }

    int laneid = tid % 32;
    int warpid = tid / 32;

    for(int i=16;i;i>>=1){
        sum += __shfl_down_sync(0xFFFFFFFF,sum,i);
    }
    __syncthreads();
    if(laneid==0){
        share[warpid]=sum;
    }
    __syncthreads();
    if(warpid==0){
        sum = tid<(blockDim.x+31/32)?share[tid]:0;
        for(int i=16;i;i>>=1){
            sum += __shfl_down_sync(0xFFFFFFFF,sum,i);
        }
    }
    if(tid==0){
        share[0]=sum;
    }
    __syncthreads();
    sum = share[0];
}
}

namespace kernel {

template <int32_t BLOCK_DIM>
__global__ void rmsnorm_kernel_cu_fp32(const float* in, const float* wei, float* out,
                                       const int size, const float eps) {

}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       tensor::Tensor& output,const float eps, void* stream) {
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
    if(stream){
        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        rmsnorm_kernel_cu_fp32<256><<<1,threads_num,0,_stream>>>(in,wei,out,size,eps);
    }else{
        rmsnorm_kernel_cu_fp32<256><<<1,threads_num>>>(in, wei, out, size, eps);
    }
}
} // namespace kernel