#include "rmsnorm_kernel.h"
#include <cmath>


namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        tensor::Tensor& output, const float eps,void*) {
    // float* in_ptr = const_cast<float*>(input.ptr<float>());
    // float* out_ptr = const_cast<float*>(output.ptr<float>());
    // float* weight_ptr=const_cast<float*>(weight.ptr<float>());
    // int size = static_cast<int32_t>(input.size());

    // float sum = 0.f;
    // for(int i = 0;i<size;i++){
    //     float input_value = in_ptr[i];
    //     sum += input_value*input_value;
    // }

    // const float eps = 1e-5f;
    // float mean = sum / float(size)+eps;

    // const float rsqrt=1.f/std::sqrt(mean);

    // for(int i=0;i<size;i++){
    //     out_ptr[i]=rsqrt*weight_ptr[i]*in_ptr[i];
    
    float* in = const_cast<float*>(input.ptr<float>());
    float* wei = const_cast<float*>(weight.ptr<float>());
    float* out = const_cast<float*>(output.ptr<float>());

    size_t sz = input.size();
    float sum=0.f;
    
    for(int i=0;i<sz;i++){
        float in_val=in[i];
        sum+=in_val*in_val;
    }
    const float rsqrt = 1.f / std::sqrt(sum/float(sz)+eps);

    for(int i=0;i<sz;i++){
        out[i]=in[i]*wei[i]*rsqrt;
    }
}
} // namespace kernel