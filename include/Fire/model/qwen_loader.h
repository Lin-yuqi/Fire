#pragma once

#include "Fire/base/base.h"
#include "Fire/model/fire_reader.h"
#include "Fire/model/model_weights.h"
#include "Fire/op/operator.h"

#include <string>
#include <vector>

namespace model {

// Adapter from the canonical .fire tensor names to one validated Qwen3Weights
// value. The selected profile defines every expected tensor shape.
class Qwen3Loader {
  public:
    explicit Qwen3Loader(Qwen3Profile profile);

    const Qwen3Profile& profile() const noexcept;
    base::Status open(const std::string& path);

    // Low-level lookup without Qwen3 profile validation. The returned CPU view
    // shares ownership of the file mapping.
    base::Status loader_tensor(const std::string& name, tensor::Tensor& output_tensor) const;

    // Validates complete FP32, INT4, or mixed BF16/INT4 profiles before publishing output.
    base::Status load_weights(Qwen3Weights& output_weights) const;

  private:
    base::Status load_tensor(const std::string& name, const std::vector<int32_t>& expected_dims,
                             tensor::Tensor& output_tensor,
                             base::DataType expected_dtype = base::DataType::Fp32) const;

    base::Status load_tensor(const std::string& name, const std::vector<int32_t>& expected_dims,
                             op::Parameter& output_parameter) const;

    Qwen3Profile _profile;
    FireReader _reader;
};

} // namespace model
