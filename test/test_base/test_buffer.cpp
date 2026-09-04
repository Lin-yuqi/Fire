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
    Buffer buf(ptr,32*sizeof(float),base::DeviceType::CPU);
    ASSERT_EQ(buf.owns_memory(),false);
    delete[] ptr;
}

TEST(allocator_test,gpu_factory_reports_gpu_device){
    auto allocator = base::GPUAllocatorFactory::get_instance();

    ASSERT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->device_type(), base::DeviceType::GPU);
}
