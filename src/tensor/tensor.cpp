#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "Fire/base/buffer.h"
#include <Fire/tensor/tensor.h>
#include <cstdint>
#include <glog/logging.h>
#include <numeric>

// ----------------tensor begin----------------
namespace tensor {  

    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    inline size_t size_of_dtype(base::DataType dtype){
        switch(dtype){
            case base::DataType::Fp32:
                return 4;
            case base::DataType::int32:
                return 4;
            case base::DataType::int8:
                return 1;
            default:
                LOG(FATAL) << "Unsupported data type: " << static_cast<int>(dtype);
                return 0;
        }   
    }

    // 只创建 Tensor 元数据，不分配内存
    Tensor::Tensor(base::DataType dtype, std::vector<int32_t> dims)
    :_data_type(dtype), _dims(dims)
    {
        update_shape_info();
    }

    // 创建 Tensor，并通过 allocator 分配内存
    Tensor::Tensor(base::DataType dtype,
           std::vector<int32_t> dims,
           std::shared_ptr<base::DeviceAllocator> allocator)
    : _data_type(dtype), _dims(dims)       
    {
        update_shape_info();
        allocate(std::move(allocator));
    }

    // 使用已有 Buffer
    Tensor::Tensor(base::DataType dtype,
           std::vector<int32_t> dims,
           std::shared_ptr<base::Buffer> buffer,
           size_t offset)
    : _data_type(dtype), _dims(dims)
    {   
        update_shape_info();
        assign(std::move(buffer), offset);
    }

    // 包装外部内存
   Tensor Tensor::from_blob(
        void* ptr,
        base::DataType dtype,
        std::vector<int32_t> dims,
        base::DeviceType device_type)
    {
        Tensor tensor(dtype, dims);
        size_t sz = tensor.byte_size();
        auto buffer= std::make_shared<base::Buffer>(sz,nullptr,ptr,true);
        tensor.set_device_type(device_type);
        buffer->set_device_type(device_type);
        tensor.assign(buffer, 0);
        return tensor;
    }

    size_t Tensor::size() const{
        return _size;
    }
    size_t Tensor::byte_size() const{
        return _size * size_of_dtype(_data_type);
    }

    int32_t Tensor::dims_size() const{
        return static_cast<int32_t>(_dims.size());
    }
    int32_t Tensor::get_dim(int32_t idx) const{
        CHECK_GE(idx, 0) << "Dimension index must be non-negative";
        CHECK_LT(idx, static_cast<int32_t>(_dims.size())) << "Dimension index out of range";
        return _dims[idx];
    }

    const std::vector<int32_t>& Tensor::dims() const{
        return _dims;
    }
    const std::vector<int64_t>& Tensor::strides() const{
        return _strides;
    }

    base::DataType Tensor::data_type() const{
        return _data_type;
    }
    base::DeviceType Tensor::device_type() const{
        if(_buffer){
            return _buffer->device_type();
        }else{
            return base::DeviceType::Unknown;
        }
    }

    void Tensor::set_device_type(base::DeviceType device_type) const{
        if(_buffer){
            _buffer->set_device_type(device_type);
        }
    }

    bool Tensor::is_empty() const{
        return _size == 0 || _buffer == nullptr || _buffer->ptr() == nullptr;
    }


    void Tensor::reshape(const std::vector<int32_t>& dims){
        CHECK(!dims.empty())<< "Tensor dims cannot be empty";
        size_t new_size = 1;
        for (int32_t dim : dims) {
            CHECK_GT(dim, 0)<< "Tensor dimension must be greater than 0";
            new_size *= static_cast<size_t>(dim);
        }
        CHECK_EQ(new_size, _size)
            << "Cannot reshape tensor from "
            << _size << " elements to "
            << new_size << " elements";
        _dims = dims;
        update_shape_info();
    }

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator){
        CHECK_NE(allocator, nullptr) << "Allocator cannot be null";
        size_t sz=this->byte_size();
        CHECK_GT(sz, 0) << "Tensor size must be greater than zero";

        _buffer = std::make_shared<base::Buffer>(sz, allocator);
        return true;
    }

    void Tensor::assign(std::shared_ptr<base::Buffer> buffer, size_t offset){
        CHECK_NE(buffer, nullptr) << "Buffer cannot be null";
        if(_buffer){
            CHECK_EQ(_buffer->device_type(), buffer->device_type()) << "Buffer device type mismatch";
        }
        size_t sz=this->byte_size();    
        CHECK_LE(sz, buffer->size() - offset)<< "Buffer space is too small for tensor";
        _buffer = std::move(buffer);
        _offset = offset;
    }

    Tensor Tensor::clone() const {
        CHECK_NE(_buffer, nullptr)<< "Cannot clone tensor without buffer";

        CHECK_NE(_buffer->ptr(), nullptr)<< "Cannot clone empty buffer";
        std::shared_ptr<base::DeviceAllocator> allocator;
        const auto device = device_type();
        if (device == base::DeviceType::CPU) {
            allocator =base::CPUAllocatorFactory::get_instance();
        }
        else if (device == base::DeviceType::GPU) {
            allocator =
                base::GPUAllocatorFactory::get_instance();
        }
        else {
            LOG(FATAL)
                << "Cannot clone tensor with unknown device";
        }

        // 创建一个新的独立 Tensor
        Tensor result(
            _data_type,
            _dims,
            allocator
        );

        const size_t bytes = byte_size();

        if (device == base::DeviceType::CPU) {
            allocator->memcpy(
                raw_ptr(),
                result.raw_ptr(),
                bytes,
                base::MemCpyKind::CPU2CPU
            );
        }
        else {
            allocator->memcpy(
                raw_ptr(),
                result.raw_ptr(),
                bytes,
                base::MemCpyKind::GPU2GPU
            );
        }

        return result;
    }

    void Tensor::to_cpu(){
        CHECK_NE(_buffer, nullptr) << "Buffer is null, cannot transfer to CPU";
        const base::DeviceType current_device = _buffer->device_type();
        if (current_device == base::DeviceType::CPU) {
            LOG(INFO) << "Tensor is already on CPU, no transfer needed";
        }else if(current_device == base::DeviceType::GPU) {
            // Implement the logic to transfer data from CUDA to CPU
            auto cpu_allocator = base::CPUAllocatorFactory::get_instance();

            auto cpu_buffer = std::make_shared<base::Buffer>(this->byte_size(), cpu_allocator);
            cpu_allocator->memcpy(_buffer->ptr(), cpu_buffer->ptr(), this->byte_size(),base::MemCpyKind::GPU2CPU);

            _buffer = std::move(cpu_buffer);
            // This is a placeholder for the actual implementation
            LOG(INFO) << "Transferring tensor from CUDA to CPU";
        } else {
            LOG(FATAL) << "Unsupported device type for transfer: " << static_cast<int>(current_device);
        }
    }
    void Tensor::to_cuda(cudaStream_t stream){
        CHECK_NE(_buffer, nullptr) << "Buffer is null, cannot transfer to CUDA";
        const base::DeviceType current_device = _buffer->device_type();
        if (current_device == base::DeviceType::GPU) {
            LOG(INFO) << "Tensor is already on CUDA, no transfer needed";
        } else if (current_device == base::DeviceType::CPU) {
            // Implement the logic to transfer data from CPU to CUDA
            auto gpu_allocator = base::GPUAllocatorFactory::get_instance();
            auto gpu_buffer = std::make_shared<base::Buffer>(this->byte_size(), gpu_allocator);
            gpu_allocator->memcpy(_buffer->ptr(), gpu_buffer->ptr(), this->byte_size(), base::MemCpyKind::CPU2GPU, stream);
            _buffer = std::move(gpu_buffer);
            // This is a placeholder for the actual implementation
            LOG(INFO) << "Transferring tensor from CPU to CUDA";
        } else {
            LOG(FATAL) << "Unsupported device type for transfer: " << static_cast<int>(current_device);
        }
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
        this->_data_type = data_type;
        this->_dims = dims;
        this-> _size = reduce_dimension(dims.begin(), dims.end(), 1);
        this-> _buffer = nullptr;
    }

    void* Tensor::raw_ptr(){
        return _buffer->ptr();
    }
    const void* Tensor::raw_ptr() const{
        return _buffer->ptr();
    }


    void Tensor::update_shape_info(){
        if (_dims.empty()) {
            _size = 0;
            _strides.clear();
            return;
        }
        _size = 1;
        for (int32_t dim : _dims) {
            CHECK_GT(dim, 0)
                << "Tensor dimension must be greater than 0";
            _size *= static_cast<size_t>(dim);
        }
        _strides.resize(_dims.size());
        int32_t stride = 1;

        for (int32_t i = static_cast<int32_t>(_dims.size()) - 1;i >= 0;--i) {
            _strides[i] = stride;
            stride *= _dims[i];
        }
    }



}