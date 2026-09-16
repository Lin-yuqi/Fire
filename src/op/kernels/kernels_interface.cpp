#include "kernels_interface.h"
#include "Fire/base/base.h"
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "cpu/rmsnorm_kernel.h"
#include "cuda/rmsnorm_kernel.cuh"
#include "cpu/matmul_kernel.h"
#include "cuda/matmul_kernel.cuh"
#include "cpu/emb_kernel.h"
#include "cuda/emb_kernel.cuh"
#include "cpu/rope_kernel.h"
#include "cuda/rope_kernel.cuh"
#include "cpu/softmax_kernel.h"
#include "cuda/softmax_kernel.cuh"
#include "cpu/swiglu_kernel.h"
#include "cuda/swiglu_kernel.cuh"

// -----------------kernel begin------------------
namespace kernel {

AddKernel get_add_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return add_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return add_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return rmsnorm_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return rmsnorm_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

RMSNormKernelDim get_rmsnorm_kernel_dim(base::DeviceType dtype) {
    if (dtype == base::DeviceType::GPU) {
        return rmsnorm_kernel_cu_dim;
    } else {
        LOG(FATAL) << "Unknown device type for get a rmsnorm kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return matmul_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return matmul_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a matmul kernel.";
        return nullptr;
    }
}

EmbeddingKernel get_embedding_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return emb_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return emb_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a emb kernel.";
        return nullptr;
    }
}

RoPEKernel get_rope_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return rope_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return rope_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

RoPECacheKernel get_rope_cache_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return sin_cos_cache_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return sin_cos_cache_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

SwiGLUKernel get_swiglu_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return swiglu_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return swiglu_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

SoftmaxKernel get_softmax_kernel(base::DeviceType dtype) {
    if (dtype == base::DeviceType::CPU) {
        return softmax_kernel_cpu;
    } else if (dtype == base::DeviceType::GPU) {
        return softmax_kernel_cu;
    } else {
        LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
        return nullptr;
    }
}

} // namespace kernel
// -----------------kernel end--------------------
