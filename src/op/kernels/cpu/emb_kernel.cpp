#include "emb_kernel.h"
#include "bf16.h"
#include "Fire/base/alloc.h"
#include <armadillo>
#include <cstdint>

// -----------------kernel begin------------------
namespace kernel {
void emb_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                    tensor::Tensor& output, void*) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);

    CHECK(weight.device_type() == output.device_type());
    CHECK(input.device_type() == base::DeviceType::CPU);
    int32_t vocab_size = weight.get_dim(0);
    auto alloc = base::CPUAllocatorFactory::get_instance();
    for(int i=0;i<input_num;i++){
        int32_t token_id = input.ptr<int32_t>()[i];
        if(token_id >= vocab_size){
            LOG(FATAL)<<"error token_id";
        }else{
            float* dst_ptr = &output.ptr<float>()[i*weight_dim];
            if (weight.data_type() == base::DataType::Bf16) {
                const uint16_t* src_ptr =
                    weight.ptr<uint16_t>() + static_cast<size_t>(token_id) * weight_dim;
                for (int32_t col = 0; col < weight_dim; ++col) {
                    dst_ptr[col] = bf16_to_fp32(src_ptr[col]);
                }
            } else {
                float* src_ptr = const_cast<float*>(&weight.ptr<float>()[token_id*weight_dim]);
                alloc->memcpy(src_ptr, dst_ptr, weight_dim*sizeof(float));
            }
        }
    }


}

} // namespace kernel
