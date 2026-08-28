#include <Fire/base/alloc.h>
#include <cstdlib>

//-----------------base begin------------------
namespace base {

CPUAllocator::CPUAllocator():DeviceAllocator(DeviceType::CPU){}

void CPUAllocator::release(void* ptr) const{
    free(ptr);
}
    
void* CPUAllocator::allocate(size_t sz)const {
    if(sz==0)return nullptr;
    void* p= malloc(sz);
    CHECK_NE(p, nullptr)<<"alloc error\n";
    return p;
}


}// --------------------base end----------------
