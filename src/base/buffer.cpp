#include "Fire/base/base.h"
#include <Fire/base/buffer.h>

// ----------------base begin--------------
namespace base {

 Buffer::Buffer(size_t capacity_bytes,std::shared_ptr<DeviceAllocator> allocator)
 :_capacity_bytes(capacity_bytes),_allocator(allocator),_owns_memory(true)
 {
    CHECK(_allocator != nullptr);
    CHECK_GT(_capacity_bytes, 0);

    _device_type = _allocator->device_type();

    _ptr = _allocator->allocate(_capacity_bytes);

    CHECK(_ptr != nullptr)
        << "Failed to allocate buffer";
 }

Buffer::Buffer(void* ptr,size_t capacity_bytes,DeviceType device_type)
:_ptr(ptr),_capacity_bytes(capacity_bytes),_device_type(device_type),_owns_memory(false)
{
    CHECK(_ptr != nullptr);
    CHECK_GT(_capacity_bytes, 0);
}


Buffer::~Buffer(){
    if (_owns_memory && _ptr) {
        CHECK(_allocator != nullptr);
        _allocator->release(_ptr);
        _ptr = nullptr;
    }
}

void* Buffer::ptr(){
    return _ptr;
}

const void* Buffer::ptr() const{
    return _ptr;
}

DeviceType Buffer::device_type(){
    return _device_type;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator(){
    return _allocator;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this(){
    return get_shared_from_this();
}
bool Buffer::owns_memory() const{
    return _owns_memory;
}

size_t Buffer::size() const{
    return _capacity_bytes;
}
void Buffer::set_device_type(DeviceType dtype){
    _device_type=dtype;
}

}// -------------base end-----------------

