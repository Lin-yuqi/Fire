#pragma once
#include "Fire/base/base.h"
#include "alloc.h"
#include <cstddef>
#include <memory>

// ---------------------- base begin ---------------
namespace base { 



class Buffer : public std::enable_shared_from_this<Buffer> , NoCopyable{
public:
    explicit Buffer()=default;

    explicit Buffer(size_t sz,std::shared_ptr<DeviceAllocator> alloc=nullptr,void* ptr=nullptr,bool use_external=false);

    virtual ~Buffer();

    bool alloc();

    //这个copy_from函数定位不是很清晰，暂时不做实现
    // void copy_from(const Buffer* buffer)const;

    // void copy_from(const Buffer& buffer)const;

    void* ptr();

    const void* ptr() const;

    DeviceType device_type();

    std::shared_ptr<DeviceAllocator> allocator();

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;

    size_t size() const;

    void set_dtype(DeviceType dtype);

private:
    size_t _byte_sz=0;
    void* _ptr=nullptr;
    bool _use_external=false;
    DeviceType _device_type = DeviceType::Unknown;
    std::shared_ptr<DeviceAllocator> _allocator;
};




}// --------------------- base end ------------------
