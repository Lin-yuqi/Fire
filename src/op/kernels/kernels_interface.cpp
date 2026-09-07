#include "kernels_interface.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"

// -----------------kernel begin------------------
namespace kernel {

AddKernel get_add_kernel(base::DeviceType dtype){
    if(dtype==base::DeviceType::CPU){
        return add_kernel_cpu;
    }else if(dtype==base::DeviceType::GPU){
        return add_kernel_cu;
    }else{
        LOG(FATAL)<<"Unknown device type for get a add kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType dtype){
    if(dtype==base::DeviceType::CPU){
        return rmsnorm_kernel_cpu;
    }else if(dtype==base::DeviceType::GPU){
        return rmsnorm_kernel_cu;
    }else{
        LOG(FATAL)<<"Unknown device type for get a add kernel.";
        return nullptr;
    }
}


}
// -----------------kernel end--------------------