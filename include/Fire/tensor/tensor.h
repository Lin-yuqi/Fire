#pragma once
#include "Fire/base/alloc.h"
#include "Fire/base/buffer.h"
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>


// -------------------tensor begin-------------------
namespace tensor{

class Tensor {
public:
    Tensor() = default;

    // 只创建 Tensor 元数据，不分配内存
    Tensor(base::DataType dtype, std::vector<int32_t> dims);

    // 创建 Tensor，并通过 allocator 分配内存
    Tensor(base::DataType dtype,
           std::vector<int32_t> dims,
           std::shared_ptr<base::DeviceAllocator> allocator);

    // 使用已有 Buffer
    Tensor(base::DataType dtype,
           std::vector<int32_t> dims,
           std::shared_ptr<base::Buffer> buffer,
           size_t offset = 0);

    // 包装外部内存
    static Tensor from_blob(
        void* ptr,
        base::DataType dtype,
        std::vector<int32_t> dims,
        base::DeviceType device_type);

public:
    size_t size() const;
    size_t byte_size() const;

    int32_t dims_size() const;
    int32_t get_dim(int32_t idx) const;

    const std::vector<int32_t>& dims() const;
    const std::vector<int64_t>& strides() const;

    base::DataType data_type() const;
    base::DeviceType device_type() const;
    void set_device_type(base::DeviceType device_type) const;

    bool is_empty() const;

    template <typename T>
    T* ptr();

    template <typename T>
    const T* ptr() const;

    template <typename T>
    T* ptr(size_t index);

    void reshape(const std::vector<int32_t>& dims);

    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator);

    void assign(std::shared_ptr<base::Buffer> buffer,
                size_t offset = 0);

    Tensor clone() const;

    void to_cpu();
    void to_cuda(cudaStream_t stream = nullptr);

    void reset(base::DataType data_type, const std::vector<int32_t>& dims);

private:
    void update_shape_info();

private:
    base::DataType _data_type =base::DataType::Unknown;
    std::vector<int32_t> _dims;
    std::vector<int64_t> _strides;
    size_t _size = 0;
    // 元素 offset，而不是 byte offset
    size_t _offset = 0;
    std::shared_ptr<base::Buffer> _buffer;
};

}// -------------------tensor end---------------------
