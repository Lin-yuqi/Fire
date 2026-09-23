#include "Fire/op/linear.h"
#include "Fire/op/operator.h"
#include "Fire/tensor/tensor.h"
#include "kernels/kernels_interface.h"

namespace op {

LinearOp::LinearOp() : ParamOperator(OpType::Linear) {}
LinearOp::LinearOp(float matmulop) : ParamOperator(OpType::Linear), _matmulop(matmulop) {}

base::Status LinearOp::forward(const tensor::Tensor& input, tensor::Tensor& output,
                               const OpContext& context) {
    auto status = _check(input, output, context);
    if (!status)
        return status;

    auto dtype = context._device_type;
    auto& weight = _params[0]._data;
    if (_params[0].is_quantized()) {
        auto& sacles = _params[0]._scales;
        auto& group_size = _params[0]._quant_config._group_size;
        auto& zero_points = _params[0]._zero_points;

        kernel::get_matmul_quant_kernel(dtype)(input, weight, output, group_size, sacles,
                                               zero_points, context._stream);

    } else {

        kernel::get_matmul_kernel(dtype)(input, weight, _matmulop, output, context._stream);
        if (_params.size() == 2) {
            auto& bias = _params[1]._data;
            kernel::get_add_kernel(dtype)(output, bias, output, context._stream);
        }
    }
    return base::error::Success();
}

base::Status LinearOp::_check(const tensor::Tensor& input, tensor::Tensor& output,
                              const OpContext& context) {
    const auto device_type = context._device_type;
    if (device_type != base::DeviceType::CPU && device_type != base::DeviceType::GPU) {
        return base::error::InvalidArgument("linear requires a CPU or GPU device");
    }
    if (_params.empty() || _params.size() > 2) {
        return base::error::InvalidArgument("linear requires a weight and an optional bias");
    }

    auto status = _check_tensor(input, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }
    status = _check_tensor(output, device_type, base::DataType::Fp32);
    if (!status) {
        return status;
    }

    if (input.dims_size() != 1 && input.dims_size() != 2) {
        return base::error::InvalidArgument("linear only supports one- or two-dimensional input");
    }
    if (output.dims_size() != input.dims_size()) {
        return base::error::InvalidArgument("linear input and output rank mismatch");
    }

    const auto& weight = _params[0]._data;
    int32_t output_features = 0;
    const int32_t input_features = input.get_dim(input.dims_size() - 1);

    if (_params[0].is_quantized()) {
        if (device_type != base::DeviceType::GPU) {
            return base::error::InvalidArgument("quantized linear requires a GPU device");
        }
        if (_params.size() != 1) {
            return base::error::InvalidArgument("quantized linear does not support bias");
        }
        if (_params[0]._quant_config._quant_type != QuantType::Int4GroupWise) {
            return base::error::InvalidArgument("linear only supports group-wise int4 weights");
        }

        status = _check_tensor(weight, device_type, base::DataType::UInt8);
        if (!status) {
            return status;
        }
        if (weight.dims_size() != 2) {
            return base::error::InvalidArgument("quantized linear weight must be two-dimensional");
        }

        output_features = weight.get_dim(0);
        if (input_features % 2 != 0 || weight.get_dim(1) != input_features / 2) {
            return base::error::InvalidArgument(
                "linear input feature dimension does not match packed int4 weight");
        }

        const int32_t group_size = _params[0]._quant_config._group_size;
        if (group_size <= 0 || input_features % group_size != 0) {
            return base::error::InvalidArgument(
                "quantized linear group size must be positive and divide input features");
        }

        const std::vector<int32_t> expected_metadata_dims{
            output_features, input_features / group_size};
        const auto& scales = _params[0]._scales;
        status = _check_tensor(scales, device_type, base::DataType::Fp32);
        if (!status) {
            return status;
        }
        if (scales.dims() != expected_metadata_dims) {
            return base::error::InvalidArgument("quantized linear scales shape mismatch");
        }

        const auto& zero_points = _params[0]._zero_points;
        status = _check_tensor(zero_points, device_type, base::DataType::UInt8);
        if (!status) {
            return status;
        }
        if (zero_points.dims() != expected_metadata_dims) {
            return base::error::InvalidArgument("quantized linear zero points shape mismatch");
        }
    } else {
        status = _check_tensor(weight, device_type, base::DataType::Fp32);
        if (!status) {
            return status;
        }
        if (weight.dims_size() != 2) {
            return base::error::InvalidArgument("linear weight must be two-dimensional");
        }

        output_features = weight.get_dim(0);
        if (weight.get_dim(1) != input_features) {
            return base::error::InvalidArgument(
                "linear input feature dimension does not match weight");
        }

        if (_params.size() == 2) {
            const auto& bias = _params[1]._data;
            status = _check_tensor(bias, device_type, base::DataType::Fp32);
            if (!status) {
                return status;
            }
            if (bias.size() != output.size()) {
                return base::error::InvalidArgument(
                    "linear bias size must match the output size");
            }
        }
    }

    if (output.get_dim(output.dims_size() - 1) != output_features) {
        return base::error::InvalidArgument(
            "linear output feature dimension does not match weight");
    }
    if (input.dims_size() == 2 && input.get_dim(0) != output.get_dim(0)) {
        return base::error::InvalidArgument("linear input and output row count mismatch");
    }

    return base::error::Success();
}

} // namespace op
