#include "Fire/base/alloc.h"
#include "Fire/base/buffer.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <Fire/base/base.h>

TEST(buffer_test,allocate){
    using namespace base;
    auto alloc= CPUAllocatorFactory::get_instance();

    Buffer buf(256,alloc);
    CHECK_NE(buf.ptr(),nullptr);
}

TEST(buffer_test,external){
    using namespace base;
    
    float* ptr= new float[32];
    Buffer buf(32*sizeof(float),nullptr,ptr,true);
    ASSERT_EQ(buf.is_external(),true);
    delete[] ptr;
}