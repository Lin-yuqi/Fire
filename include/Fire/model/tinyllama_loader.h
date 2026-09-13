#include "Fire/base/base.h"
#include <Fire/tensor/tensor.h>
#include <Fire/model/fire_reader.h>

namespace model {

class TinyllamaLoader {
  public:
    TinyllamaLoader()=default;

    base::Status open(const std::string& path);

    base::Status loader_tensor(const std::string& name, tensor::Tensor& output_tensor) const;

  private:
    FireReader _reader;
};

} // namespace model