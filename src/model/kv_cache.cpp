#include "Fire/base/base.h"
#include "Fire/tensor/tensor.h"
#include <Fire/model/kv_cache.h>
#include <cstddef>

namespace model {

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