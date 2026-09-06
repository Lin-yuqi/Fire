#include "kernels_interface.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"

// -----------------kernel begin------------------
namespace kernel {

AddKernel get_add_kernel(base::DeviceType dtpye){
    if(dtpye==base::DeviceType::CPU){
        return add_kernel_cpu;
    }else if(dtpye==base::DeviceType::GPU){
        return add_kernel_cu;
    }else{
        LOG(FATAL)<<"Unknown device type for get a add kernel.";
        return nullptr;
    }
}


}
// -----------------kernel end--------------------