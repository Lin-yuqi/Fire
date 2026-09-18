#include "Fire/base/alloc.h"
#include "Fire/base/base.h"
#include "Fire/tensor/tensor.h"
#include <Fire/model/kv_cache.h>
#include <cstddef>
#include <cstring>
#include <vector>

namespace model {
// key_cache: [num_layers, capacity, num_kv_heads, head_dim]
// val_cache: [num_layers, capacity, num_kv_heads, head_dim]
base::Status KVCache::allocate(int32_t num_layers, int32_t capacity, int32_t num_kv_heads,
                               int32_t head_dim, std::shared_ptr<base::DeviceAllocator> allocator) {
    if (num_layers <= 0 || capacity <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        allocator == nullptr) {
        return base::error::InvalidArgument("KVCache::allocate invalid argument");
    }

    _key = tensor::Tensor(base::DataType::Fp32, {num_layers, capacity, num_kv_heads, head_dim},
                          allocator);
    _value = tensor::Tensor(base::DataType::Fp32, {num_layers, capacity, num_kv_heads, head_dim},
                            allocator);
    _num_kv_heads = num_kv_heads;
    _num_layers = num_layers;
    _head_dim = head_dim;
    _length = 0;
    _capacity = capacity;
    return base::error::Success();
}

base::Status KVCache::write(int32_t layer_idx, int32_t pos, const tensor::Tensor& cur_k,
                            const tensor::Tensor& cur_v, cudaStream_t stream) {
    if (layer_idx < 0 || layer_idx >= _num_layers) {
        return base::error::InvalidArgument("KVCache layer index is out of range");
    }
    if (pos < 0 || pos >= _capacity) {
        return base::error::InvalidArgument("KVCache position is out of range");
    }
    if (cur_k.device_type() != _key.device_type() ||
        cur_v.device_type() != _value.device_type()) {
        return base::error::InvalidArgument("KVCache input device does not match cache device");
    }
    if (cur_k.data_type() != base::DataType::Fp32 ||
        cur_v.data_type() != base::DataType::Fp32 ||
        cur_k.dims() != std::vector<int32_t>{_num_kv_heads, _head_dim} ||
        cur_v.dims() != std::vector<int32_t>{_num_kv_heads, _head_dim}) {
        return base::error::InvalidArgument(
            "KVCache inputs must be FP32 [num_kv_heads, head_dim]");
    }

    size_t cache_offset = offset(layer_idx, pos);
    size_t bytes = static_cast<size_t>(_num_kv_heads) * _head_dim * sizeof(float);
    float* key_dst = _key.ptr<float>() + cache_offset;
    float* val_dst = _value.ptr<float>() + cache_offset;

    if (_key.device_type() == base::DeviceType::CPU) {
        std::memcpy(key_dst, cur_k.ptr<void>(), bytes);
        std::memcpy(val_dst, cur_v.ptr<void>(), bytes);
    } else if (_key.device_type() == base::DeviceType::GPU) {
        auto cuda_status =
            cudaMemcpyAsync(key_dst, cur_k.ptr<void>(), bytes, cudaMemcpyDeviceToDevice, stream);
        if (cuda_status != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(cuda_status));
        }
        cuda_status =
            cudaMemcpyAsync(val_dst, cur_v.ptr<void>(), bytes, cudaMemcpyDeviceToDevice, stream);
        if (cuda_status != cudaSuccess) {
            return base::error::InternalError(cudaGetErrorString(cuda_status));
        }
    } else {
        return base::error::InvalidArgument("KVCache requires a CPU or GPU device");
    }
    return base::error::Success();
}

base::Status KVCache::commit(int32_t pos) {
    if (pos != _length || pos < 0 || pos >= _capacity) {
        return base::error::InvalidArgument(
            "KVCache commit position does not match the current length");
    }
    ++_length;
    return base::error::Success();
}

void KVCache::reset() { _length = 0; }

int32_t KVCache::capacity() const { return _capacity; }
int32_t KVCache::length() const { return _length; }

tensor::Tensor& KVCache::key() { return _key; }
tensor::Tensor& KVCache::value() { return _value; }

const tensor::Tensor& KVCache::key() const { return _key; }
const tensor::Tensor& KVCache::value() const { return _value; }

size_t KVCache::offset(int32_t layer, int32_t pos) const {
    return static_cast<size_t>((layer * _capacity + pos) * _num_kv_heads * _head_dim);
}

} // namespace model
