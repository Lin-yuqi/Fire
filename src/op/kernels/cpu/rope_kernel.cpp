#include "rope_kernel.h"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kernel {

void sin_cos_cache_kernel_cpu(int head_size, int max_seq_len, float rope_theta,
                              tensor::Tensor& sin_cache, tensor::Tensor& cos_cache, void*) {
    const int half_size = head_size / 2;
    float* sin = sin_cache.ptr<float>();
    float* cos = cos_cache.ptr<float>();
    for (int d = 0; d < half_size; ++d) {
        const float freq =
            1.0f / std::pow(rope_theta, 2.0f * static_cast<float>(d) / head_size);
        for (int pos = 0; pos < max_seq_len; ++pos) {
            const float theta = freq * static_cast<float>(pos);
            const size_t offset = static_cast<size_t>(pos) * half_size + d;
            sin[offset] = std::sin(theta);
            cos[offset] = std::cos(theta);
        }
    }
}

void rope_kernel_cpu(tensor::Tensor& input_q, tensor::Tensor& input_k, const tensor::Tensor& cos,
                     const tensor::Tensor& sin, int32_t pos, void*) {
    int head_size = input_q.get_dim(1);
    const int half_size = head_size / 2;
    const size_t offset = static_cast<size_t>(pos) * half_size;
    const float* sin_row = sin.ptr<float>() + offset;
    const float* cos_row = cos.ptr<float>() + offset;

    for (int32_t head = 0; head < input_q.get_dim(0); ++head) {
        float* q = input_q.ptr<float>() + head * head_size;
        for (int32_t d = 0; d < half_size; ++d) {
            const float fci = sin_row[d];
            const float fcr = cos_row[d];
            const float q0 = q[d];
            const float q1 = q[d + half_size];
            q[d] = q0 * fcr - q1 * fci;
            q[d + half_size] = q0 * fci + q1 * fcr;

            if (head < input_k.get_dim(0)) {
                float* k = input_k.ptr<float>() + head * head_size;
                const float k0 = k[d];
                const float k1 = k[d + half_size];
                k[d] = k0 * fcr - k1 * fci;
                k[d + half_size] = k0 * fci + k1 * fcr;
            }
        }
    }
}
} // namespace kernel
