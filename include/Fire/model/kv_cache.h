#pragma once
#include "Fire/base/base.h"
#include "Fire/base/alloc.h"
#include "Fire/tensor/tensor.h"
#include <cstdint>

namespace model {
class KVCache {
  public:
    KVCache() = default;

    base::Status allocate(int32_t num_layers, int32_t capacity, int32_t num_kv_heads,
                          int32_t head_dim, std::shared_ptr<base::DeviceAllocator> allocator);

    base::Status write(int32_t layer_idx, int32_t pos, const tensor::Tensor& cur_k,
                       const tensor::Tensor& cur_v, cudaStream_t stream = nullptr);
    base::Status commit(int32_t pos);
    void reset();

    int32_t capacity() const;
    int32_t length() const;

    tensor::Tensor& key();
    tensor::Tensor& value();

    const tensor::Tensor& key() const;
    const tensor::Tensor& value() const;

  private:
    size_t offset(int32_t layer, int32_t pos) const;

  private:
    // key_cache: [num_layers, capacity, num_kv_heads, head_dim]
    // val_cache: [num_layers, capacity, num_kv_heads, head_dim]
    tensor::Tensor _key;
    tensor::Tensor _value;

    /*这些成员变量方便算offset*/
    int32_t _num_layers = 0;
    int32_t _num_kv_heads = 0;
    int32_t _head_dim = 0;

    /* 这个capcity指当前的kv_cache是指目
    前这个kv_cache预留了多少token的size*/
    int32_t _capacity = 0;
    int32_t _length = 0;
};

} // namespace model
