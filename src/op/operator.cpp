#include "Fire/base/base.h"
#include <Fire/op/operator.h>
#include <cstddef>
#include <utility>

// ----------------------op begin-------------------
namespace op{
Operator::Operator(OpType type, std::string name) : _type(type), _name(std::move(name)) {}

OpType Operator::type() const {
    return _type;
}

const std::string& Operator::name() const{
    return _name;
}

void Operator::set_name(const std::string& name){
    _name=name;
}

base::Status Operator::_check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                                    base::DataType data_type) const{
    if(tensor.is_empty()){
        return base::error::InvalidArgument("the tensor is empty");
    }else if(tensor.device_type()!=device_type){
        return base::error::InvalidArgument("the tensor has a wrong device type");
    }else if(tensor.data_type()!=data_type){
        return base::error::InvalidArgument("the tensor has a wrong data type");
    }
    return base::error::Success();
}
base::Status Operator::_check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                base::DataType data_type, std::initializer_list<int32_t>expected_dims) const{
    if(tensor.is_empty()){
        return base::error::InvalidArgument("the tensor is empty");
    }else if(tensor.device_type()!=device_type){
        return base::error::InvalidArgument("the tensor has a wrong device type");
    }else if(tensor.data_type()!=data_type){
        return base::error::InvalidArgument("the tensor has a wrong data type");
    }

    const auto& dims=tensor.dims();
    if(dims.size()!=expected_dims.size()){
        return base::error::InvalidArgument("the tensor has a wrong dims size");
    }

    size_t i=0;
    for(int32_t dim:expected_dims){
        if(dims[i]!=dim){
            return base::error::InvalidArgument("the tensor dim mismatch");
        }
        ++i;
    }
    return base::error::Success();
}



size_t ParamOperator::param_size() const {
    return _params.size();
}

void ParamOperator::reset_param_size(size_t size) {
    _params.resize(size);
}

Parameter& ParamOperator::get_param(size_t idx) {
    CHECK_LT(idx, _params.size());
    return _params[idx];
}

const Parameter& ParamOperator::get_param(size_t idx) const {
    CHECK_LT(idx, _params.size());
    return _params[idx];
}

void ParamOperator::set_param(size_t idx,const Parameter& param) {
    CHECK_LT(idx, _params.size());
    _params[idx] = param;
}


}
// ----------------------op end---------------------
