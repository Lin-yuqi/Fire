#include <Fire/op/layer.h>

// -------------------op begin-------------------
namespace op {

    BaseLayer::BaseLayer(LayerType layertype,base::DataType datatype,base::DeviceType devicetype,std::string name)
    : _layer_type(layertype),_data_type(datatype),_device_type(devicetype),_layer_name(name)
    {

    }

    const std::string& BaseLayer::get_layer_name(){
        return _layer_name;
    }
    void BaseLayer::set_layer_name(const std::string& name){
        _layer_name=name;
    }

    LayerType BaseLayer::get_layer_type(){
        return _layer_type;
    }
    void BaseLayer::set_layer_type(LayerType type){
        _layer_type=type;
    }
    
    base::DataType BaseLayer::get_data_type(){
        return _data_type;
    }
    void BaseLayer::set_data_type(base::DataType type){
        _data_type=type;
    }

    base::DeviceType BaseLayer::get_device_type(){
        return _device_type;
    }
    void BaseLayer::set_device_type(base::DeviceType type){
        _device_type=type;
    }

}// -------------------op end-------------------
