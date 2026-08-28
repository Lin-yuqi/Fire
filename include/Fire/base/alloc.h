#pragma once
#include "base.h"
#include <cstddef>
#include <memory>
#include <glog/logging.h>

// ----------------base begin---------------
namespace base{

class DeviceAllocator{
public:
    explicit DeviceAllocator(DeviceType dTpye):_deviceType(dTpye){}

    virtual DeviceType device_type(){ return _deviceType; }

    virtual void release(void* ptr)const =0;

    virtual void* allocate(size_t sz) const =0;

    virtual void memcpy(const void* src, void* dst,size_t sz, MemCpyKind copyKind = MemCpyKind::CPU2CPU, void* stream = nullptr, bool need_snyc =false) const;

    virtual void memset_zero(void* ptr, size_t sz, void* stream=nullptr, bool need_snyc=false) const;

private:
    DeviceType _deviceType = DeviceType::Unknown;
};



class CPUAllocator : public DeviceAllocator{
public:
    explicit CPUAllocator();
    void release(void* ptr) const override;
    void* allocate(size_t sz)const override;
};

class GPUAllocator : public DeviceAllocator{
public:
    explicit GPUAllocator();
    void release(void* ptr) const override;
    void* allocate(size_t sz)const override;
};

class CPUAllocatorFactory{
public:
    static std::shared_ptr<CPUAllocator> get_instance(){
        if(_ins==nullptr){
            _ins=std::make_shared<CPUAllocator>();
        }
        return _ins;
    }
private:
    static std::shared_ptr<CPUAllocator> _ins ;
};

class GPUAllocatorFactory{
public:
    static std::shared_ptr<GPUAllocator> get_instance(){
        if(_ins==nullptr){
            _ins=std::make_shared<GPUAllocator>();
        }
        return _ins;
    }
private:
    static std::shared_ptr<GPUAllocator> _ins;
};



}// ----------------base end-----------------