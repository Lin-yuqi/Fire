#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "Fire/op/operator.h"
#include "Fire/tensor/tensor.h"
#include <gtest/gtest.h>
#include <Fire/op/add.h>
#include "../utils.cuh"
#include <kernels_interface.h>
#include <cuda_runtime.h>


TEST(op_test,add){
    auto alloc_cu =base::GPUAllocatorFactory::get_instance();
    int32_t sz=32*129;
    tensor::Tensor t1(base::DataType::Fp32,{sz},alloc_cu);
    tensor::Tensor t2(base::DataType::Fp32,{sz},alloc_cu);
    tensor::Tensor t3(base::DataType::Fp32,{sz},alloc_cu);

    set_value_cu(t1.ptr<float>(), sz,2.f);
    set_value_cu(t2.ptr<float>(), sz,3.f);

    op::OpContext opctext;
    opctext._device_type=base::DeviceType::GPU;

    kernel::get_add_kernel(base::DeviceType::GPU)(t1,t2,t3,nullptr);
    cudaDeviceSynchronize();
    float* output=new float[sz];
    cudaMemcpy(output,t3.ptr<float>(),sz*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<sz;i++){
        ASSERT_EQ(output[i],5.f);
    }

    delete[] output;
}