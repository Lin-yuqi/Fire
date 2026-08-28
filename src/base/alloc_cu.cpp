#include <Fire/base/alloc.h>
#include <cuda_runtime_api.h>

// ---------------base begin----------------
namespace base {

void GPUAllocator::release(void* ptr) const {
    cudaError_t err = cudaFree(ptr);
    CHECK(err==cudaSuccess)<<"cuda malloc error\n";
}


void* GPUAllocator::allocate(size_t sz)const {
    if(sz==0)return nullptr;
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, sz);
    CHECK(err==cudaSuccess)<<"cuda malloc error\n";
    return ptr;
}

}// -------------base end-----------------