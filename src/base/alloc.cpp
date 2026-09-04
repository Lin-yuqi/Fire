#include <Fire/base/alloc.h>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
// -------------------base begin-------------
namespace base {

void DeviceAllocator::memcpy(const void* src, void* dst,size_t sz, MemCpyKind copyKind, void* stream, bool need_snyc) const{
    CHECK_NE(src,nullptr);
    CHECK_NE(dst,nullptr);

    cudaStream_t _stream = nullptr;
    if(stream) _stream=static_cast<cudaStream_t>(stream);

    if(copyKind==MemCpyKind::CPU2CPU){
        std::memcpy(dst, src, sz);
    }else if(copyKind==MemCpyKind::CPU2GPU){
        if(stream){
            cudaMemcpyAsync(dst,src,sz,cudaMemcpyKind::cudaMemcpyHostToDevice,_stream);
        }else{
            cudaMemcpy(dst,src,sz,cudaMemcpyKind::cudaMemcpyHostToDevice);
        }
    }else if(copyKind==MemCpyKind::GPU2GPU){
        if(stream){
            cudaMemcpyAsync(dst,src,sz,cudaMemcpyKind::cudaMemcpyDeviceToDevice,_stream);
        }else{
            cudaMemcpy(dst,src,sz,cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }
    }else if(copyKind==MemCpyKind::GPU2CPU){
        if(stream){
            cudaMemcpyAsync(dst,src,sz,cudaMemcpyKind::cudaMemcpyDeviceToHost,_stream);
        }else{
            cudaMemcpy(dst,src,sz,cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }
    }else{
        LOG(FATAL)<<"Unknown memcpy kind: "<<int(copyKind)<<"\n";
    }
    if(need_snyc){
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(void* ptr, size_t sz, void* stream, bool need_snyc) const{
    
    CHECK(_deviceType!=DeviceType::Unknown);

    cudaStream_t _stream = nullptr;
    if(stream) _stream=static_cast<cudaStream_t>(stream);
    
    if(_deviceType==DeviceType::CPU){
        memset(ptr, 0, sz);
    }else{
        if(!stream){
            cudaMemset(ptr, 0, sz);
        }else{
            cudaMemsetAsync(ptr, 0, sz, _stream);
        }
    }
    if(need_snyc){
        cudaDeviceSynchronize();
    }
}

std::shared_ptr<CPUAllocator> CPUAllocatorFactory::_ins = nullptr;
std::shared_ptr<GPUAllocator> GPUAllocatorFactory::_ins = nullptr;


}// ------------------base end---------------
