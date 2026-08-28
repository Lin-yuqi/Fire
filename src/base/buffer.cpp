#include "Fire/base/base.h"
#include <Fire/base/buffer.h>

// ----------------base begin--------------
namespace base {

Buffer::Buffer(size_t sz,std::shared_ptr<DeviceAllocator> alloc,void* ptr,bool use_external)
    :_byte_sz(sz),_ptr(ptr),_use_external(use_external),_allocator(alloc)
{
    if(_use_external&&_ptr){
        //使用外部的资源,此时ptr有空间,外部不传入alloc;
    }else{
        //自己管理资源,此时ptr为空,外部传入alloc;
        CHECK(_ptr==nullptr);
        CHECK_NE(_allocator,nullptr);
        
        _ptr = _allocator->allocate(_byte_sz);
        _device_type=_allocator->device_type();
    }
}

Buffer::~Buffer(){
    if(!_use_external){
        if(_ptr&&_allocator){
            _allocator->release(_ptr);
        }
    }
}


bool Buffer::alloc(){
    if(_allocator&&_byte_sz!=0){
        _use_external=false;
        _ptr=_allocator->allocate(_byte_sz);
        if(_ptr)return true;
        else return false;
    }else{
        return false;
    }
}


// void Buffer::copy_from(const Buffer* buffer)const{

// }

// void Buffer::copy_from(const Buffer& buffer)const{

// }

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
bool Buffer::is_external() const{
    return _use_external;
}

size_t Buffer::size() const{
    return _byte_sz;
}
void Buffer::set_dtype(DeviceType dtype){
    _device_type=dtype;
}

}// -------------base end-----------------

