#include <Fire/op/rope.h>

namespace op {
base::Status RoPEOp::forward(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                             const tensor::Tensor& sin, int32_t position,
                             const OpContext& context) {

}

base::Status RoPEOp::_check(tensor::Tensor& query, tensor::Tensor& key, const tensor::Tensor& cos,
                            const tensor::Tensor& sin, int32_t position, const OpContext& context) {
}

} // namespace op