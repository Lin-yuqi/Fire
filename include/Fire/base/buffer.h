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

    // 自己申请内存
    explicit Buffer(size_t capacity_bytes,std::shared_ptr<DeviceAllocator> allocator);
    // 引用外部内存，不拥有
    explicit Buffer(void* ptr,size_t capacity_bytes,DeviceType device_type);

    virtual ~Buffer();

    void* ptr();

    const void* ptr() const;

    DeviceType device_type();

    std::shared_ptr<DeviceAllocator> allocator();

    std::shared_ptr<Buffer> get_shared_from_this();

    bool owns_memory() const;

    size_t size() const;

    void set_device_type(DeviceType dtype);

private:

    size_t _capacity_bytes=0;
    void* _ptr=nullptr;
    bool _owns_memory = false;
    DeviceType _device_type = DeviceType::Unknown;
    std::shared_ptr<DeviceAllocator> _allocator;
};




}// --------------------- base end ------------------
