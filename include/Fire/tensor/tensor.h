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
           size_t byte_offset = 0);

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
    T* ptr(){
        if(_buffer == nullptr){
            return nullptr;
        }
        return reinterpret_cast<T*>(raw_ptr());
    }

    template <typename T>
    const T* ptr() const{
        if(_buffer == nullptr){
            return nullptr;
        }
        return reinterpret_cast<T*>(raw_ptr());
    }

    void reshape(const std::vector<int32_t>& dims);

    bool allocate(std::shared_ptr<base::DeviceAllocator> allocator);

    void assign(std::shared_ptr<base::Buffer> buffer,
                size_t byte_offset = 0);

    Tensor clone() const;

    void to_cpu();
    // 同步迁移：即使传入 stream，返回时数据也已拷贝完成
    void to_cuda(cudaStream_t stream = nullptr);

    void reset(base::DataType data_type, const std::vector<int32_t>& dims);

private:
    void* raw_ptr();
    const void* raw_ptr() const;
    void update_shape_info();

private:
    base::DataType _data_type =base::DataType::Unknown;
    std::vector<int32_t> _dims;
    std::vector<int64_t> _strides;
    size_t _size = 0;
    // 相对于 Buffer 起始地址的字节偏移
    size_t _byte_offset = 0;
    std::shared_ptr<base::Buffer> _buffer;
};

}// -------------------tensor end---------------------
