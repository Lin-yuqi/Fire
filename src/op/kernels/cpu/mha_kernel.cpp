#include "mha_kernel.h"
#include <cmath>
#include <cstdint>

#include "Fire/base/base.h"
#include "Fire/tensor/tensor.h"
#include "softmax_kernel.h"

namespace kernel {

// q:         [num_attention_heads, head_dim]
// key_cache: [num_layers, capacity, num_kv_heads, head_dim]
// val_cache: [num_layers, capacity, num_kv_heads, head_dim]
// score:     [num_attention_heads, capacity]
// output:    [num_attention_heads, head_dim]
void mha_kernel_cpu(const tensor::Tensor& input_q, const tensor::Tensor& key_cache,
                    const tensor::Tensor& val_cache, tensor::Tensor& score, tensor::Tensor& output,
                    int32_t layer_idx, int32_t pos, void*) {

    int32_t num_q_heads = input_q.get_dim(0);
    int32_t head_dim = input_q.get_dim(1);

    int32_t num_layers = key_cache.get_dim(0);
    int32_t capacity = key_cache.get_dim(1);
    int32_t num_kv_heads = key_cache.get_dim(2);
    int32_t kv_head_dim = key_cache.get_dim(3);

    CHECK(layer_idx >= 0 && layer_idx < num_layers);
    CHECK(pos >= 0 && pos < capacity);

    CHECK(head_dim == kv_head_dim);
    CHECK(num_q_heads % num_kv_heads == 0);

    CHECK(score.get_dim(0) == num_q_heads);
    CHECK(score.get_dim(1) == capacity);

    CHECK(output.get_dim(0) == num_q_heads);
    CHECK(output.get_dim(1) == head_dim);

    int32_t kv_mul = num_q_heads / num_kv_heads;

    float scale = 1.f / std::sqrt(static_cast<float>(head_dim));

    int32_t layer_offset = layer_idx * capacity * num_kv_heads * head_dim;

    // 每个 Q head 独立计算 attention
    for (int32_t h = 0; h < num_q_heads; ++h) {

        float* score_head_addr = score.ptr<float>() + h * capacity;

        const float* query_head_addr = input_q.ptr<float>() + h * head_dim;

        /*
         * GQA:
         *
         * num_q_heads = 32
         * num_kv_heads = 4
         * kv_mul = 8
         *
         * Q head 0~7   -> KV head 0
         * Q head 8~15  -> KV head 1
         * ...
         */
        int32_t kv_head = h / kv_mul;

        // --------------------------------------------------
        // 1. Q_h 与 K[0..pos] 做点积，得到 score
        // --------------------------------------------------
        for (int32_t t = 0; t <= pos; ++t) {

            int32_t cache_offset = t * num_kv_heads * head_dim + kv_head * head_dim;

            const float* key_head_addr = key_cache.ptr<float>() + layer_offset + cache_offset;

            // score[t] = q · k_t / sqrt(head_dim)
            float dot = 0.0f;
            for (int32_t d = 0; d < head_dim; ++d) {
                dot += query_head_addr[d] * key_head_addr[d];
            }
            score_head_addr[t] = dot * scale;
        }

        // --------------------------------------------------
        // 2. 对当前 head 的 score[0..pos] 做 softmax
        // --------------------------------------------------
        tensor::Tensor score_head_tensor = tensor::Tensor::from_blob(
            score_head_addr, base::DataType::Fp32, {pos + 1}, base::DeviceType::CPU);

        softmax_kernel_cpu(score_head_tensor, score_head_tensor, nullptr);

        // --------------------------------------------------
        // 3. output_h = sum_t score[t] * V_t
        // --------------------------------------------------
        float* output_head_ptr = output.ptr<float>() + h * head_dim;

        // 先清零，因为后面是累加
        for (int32_t d = 0; d < head_dim; ++d) {
            output_head_ptr[d] = 0.f;
        }

        for (int32_t t = 0; t <= pos; ++t) {

            int32_t cache_offset = t * num_kv_heads * head_dim + kv_head * head_dim;

            const float* value_head_addr = val_cache.ptr<float>() + layer_offset + cache_offset;

            float attention_weight = score_head_addr[t];

            for (int32_t d = 0; d < head_dim; ++d) {
                output_head_ptr[d] += attention_weight * value_head_addr[d];
            }
        }
    }
}

} // namespace kernel
