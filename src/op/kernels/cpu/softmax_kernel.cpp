#include "softmax_kernel.h"
#include <algorithm>
#include <cmath>

namespace kernel {

void softmax_kernel_cpu(const tensor::Tensor& inuput, tensor::Tensor& output, void*) {
    int N = 1;
    int C = 1;

    if (inuput.dims_size() == 1) {
        C = inuput.get_dim(0);
    } else {
        N = inuput.get_dim(0);
        C = inuput.get_dim(1);
    }

    const float* in = inuput.ptr<float>();
    float* out = output.ptr<float>();

    for (int i = 0; i < N; i++) {
        // 求最大值
        float mx = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            mx = std::max(mx, in[i * C + j]);
        }

        for (int j = 0; j < C; j++) {
            const float value = std::exp(in[i * C + j] - mx);
            out[i * C + j] = value;
            sum += value;
        }

        const float inv_sum = 1.0f / sum;
        for (int j = 0; j < C; j++) {
            out[i * C + j] *= inv_sum;
        }
    }
}

} // namespace kernel
