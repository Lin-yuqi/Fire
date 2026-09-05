#pragma once
#include <cstdint>
#include <Fire/base/base.h>
#include <string>
#include <Fire/tensor/tensor.h>
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

    virtual base::Status forward();
    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual base::Status check() const = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

    virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,const void* weight_ptr,
        base::DeviceType device_type = base::DeviceType::Unknown);


    const std::string& get_layer_name();
    void set_layer_name(const std::string& name);

    LayerType get_layer_type();
    void set_layer_type(LayerType type);
    
    base::DataType get_data_type();
    void set_data_type(base::DataType type);

    base::DeviceType get_device_type();
    void set_device_type(base::DeviceType type);

protected:
    std::string _layer_name; //层名
    LayerType _layer_type = LayerType::Unknown; //层类型
    base::DataType _data_type =base::DataType::Unknown; //数据类型
    base::DeviceType _device_type = base::DeviceType::Unknown;  //设备类型
    
};






}// ----------------op end----------------------