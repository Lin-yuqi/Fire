#pragma once

#include "Fire/base/base.h"
#include "Fire/model/fire_reader.h"
#include "Fire/model/model_weights.h"
#include "Fire/tensor/tensor.h"

#include <string>

namespace model {

class TinyllamaLoader {
  public:
    TinyllamaLoader() = default;

    // Container validation only; success does not establish a valid Model Profile.
    base::Status open(const std::string& path);

    // Low-level lookup retained for existing callers. Does not validate the
    // TinyLlama profile. The returned CPU view shares the mmap's lifetime.
    base::Status loader_tensor(const std::string& name, tensor::Tensor& output_tensor) const;

    // Planned complete-profile entry: validate all 201 names/dtypes/shapes, then
    // publish CPU views as one result. Failure leaves output_weights unchanged.
    // Scaffold only: currently returns FunctionUnImplement without modifying output.
    base::Status load_weights(TinyLlamaWeights& output_weights) const;

  private:
    FireReader _reader;
};

} // namespace model
