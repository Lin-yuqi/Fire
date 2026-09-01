#pragma once
#include <cstdint>
#include <Fire/base/base.h>
#include <string>
// -----------------op begin--------------------
namespace op{

enum class LayerType:uint8_t{
    Unknown = 0,
    Linear,
    Encode,
    Embedding,
    RMSNorm,
    Matmul,
    MHA,
    Softmax,
    Add,
    SwiGLU
};

class BaseLayer{
    explicit BaseLayer(LayerType layertype,base::DataType datatype,base::DeviceType devicetype,std::string name = "");

    const std::string& get_layer_name();
    void set_layer_name(const std::string& name);

    LayerType get_layer_type();
    void set_layer_type(LayerType type);
    
    base::DataType get_data_type();
    void set_data_type(base::DataType type);

    base::DeviceType get_device_type();
    void set_device_type(base::DeviceType type);

protected:
    std::string _layer_name;
    LayerType _layer_type = LayerType::Unknown;
    base::DataType _data_type =base::DataType::Unknown;
    base::DeviceType _device_type = base::DeviceType::Unknown;
    
};






}// ----------------op end----------------------