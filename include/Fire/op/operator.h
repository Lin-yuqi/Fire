#pragma once
#include <cstdint>
#include <Fire/base/base.h>
#include <initializer_list>
#include <string>
#include <Fire/tensor/tensor.h>
// -----------------op begin--------------------
namespace op{

enum class OpType:uint8_t{
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

enum class QuantType : uint8_t {
    None = 0,
    Int8GroupWise,
    Int8PerTensor,
    Int8PerChannel,
    Int4GroupWise,
};


struct QuantConfig {
    QuantType _quant_type = QuantType::None;

    // group-wise quantization使用
    int32_t _group_size = 0;

    // 是否对称量化
    bool _symmetric = true;
};


struct Parameter {
    tensor::Tensor _data;

    QuantConfig _quant_config;

    tensor::Tensor _scales;

    tensor::Tensor _zero_points;

    bool is_quantized() const {
        return _quant_config._quant_type != QuantType::None;
    }
};


/**
 * 一次算子执行所需要的运行时环境。
 *
 * 注意：
 * 这些信息属于“这一次执行”，
 * 而不是 Operator 本身。
 */
struct OpContext {
    base::DeviceType _device_type = base::DeviceType::Unknown;
    cudaStream_t _stream = nullptr;
    // 后面如果需要临时显存，可以继续加
    std::shared_ptr<base::DeviceAllocator> _allocator = nullptr;
    void* _workspace = nullptr;
    size_t _workspace_size = 0;
};


/**
 * 所有 Operator 的基类。
 */
class Operator {
public:
    explicit Operator(OpType type,std::string name = "");

    virtual ~Operator() = default;

    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;

    Operator(Operator&&) = default;
    Operator& operator=(Operator&&) = default;

public:
    OpType type() const;

    const std::string& name() const;

    void set_name(const std::string& name);

protected:
    base::Status _check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;
    base::Status _check_tensor_with_dim(const tensor::Tensor& tensor, base::DeviceType device_type,
                                     base::DataType data_type, std::initializer_list<int32_t>expected_dims) const;



protected:
    OpType _type = OpType::Unknown;

    std::string _name;
};


/**
 * 带模型参数的 Operator。
 *
 * 例如：
 *
 * Linear
 * RMSNorm
 * Embedding
 *
 */
class ParamOperator : public Operator {
public:
    using Operator::Operator;

    virtual ~ParamOperator() = default;

public:
    size_t param_size() const;

    void reset_param_size(size_t size);

    Parameter& get_param(size_t idx);

    const Parameter& get_param(size_t idx) const;

    void set_param(size_t idx,const Parameter& param);
protected:
    std::vector<Parameter> _params;
};


}// ----------------op end----------------------