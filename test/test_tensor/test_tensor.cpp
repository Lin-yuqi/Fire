#include "Fire/base/alloc.h"
#include "Fire/base/buffer.h"
#include "Fire/tensor/tensor.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>

TEST(tensor_test, from_blob_wraps_external_memory) {
    int values[] = {1, 2, 3, 4};

    tensor::Tensor tensor = tensor::Tensor::from_blob(
        values, base::DataType::int32, {4}, base::DeviceType::CPU);

    EXPECT_EQ(tensor.ptr<int>(), values);
    EXPECT_EQ(tensor.device_type(), base::DeviceType::CPU);
    EXPECT_EQ(tensor.byte_size(), sizeof(values));
}

TEST(tensor_test, byte_offset_points_inside_buffer) {
    auto allocator = base::CPUAllocatorFactory::get_instance();
    auto buffer = std::make_shared<base::Buffer>(32, allocator);

    tensor::Tensor tensor(base::DataType::int32, {2}, buffer, 8);

    EXPECT_EQ(tensor.ptr<int>(), reinterpret_cast<int*>(static_cast<char*>(buffer->ptr()) + 8));
    EXPECT_EQ(tensor.byte_size(), 8);
}

TEST(tensor_test, clone_copies_from_byte_offset) {
    auto allocator = base::CPUAllocatorFactory::get_instance();
    auto buffer = std::make_shared<base::Buffer>(16, allocator);
    auto* values = static_cast<int*>(buffer->ptr());
    values[0] = 10;
    values[1] = 20;
    values[2] = 30;
    values[3] = 40;

    tensor::Tensor view(base::DataType::int32, {2}, buffer, 8);
    tensor::Tensor clone = view.clone();
    const auto* cloned_values = clone.ptr<int>();

    EXPECT_EQ(cloned_values[0], 30);
    EXPECT_EQ(cloned_values[1], 40);
    EXPECT_NE(clone.ptr<int>(), view.ptr<int>());
}
