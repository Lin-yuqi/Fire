# Fire Model Import

本上下文统一描述外部模型参数进入 Fire 并达到可验证、可运行状态时使用的语言。

## Language

**Source Checkpoint**:
模型发布方提供、尚未适配 Fire 命名和布局语义的参数与配置集合。
_Avoid_: Fire 模型、导出文件

**Fire Tensor Container**:
扩展名为 `.fire` 的具名 tensor 容器；它保存 tensor 目录和数据，但不负责标识或描述完整模型。
_Avoid_: 自描述模型、通用 checkpoint

**Format Version**:
Fire Tensor Container 字节布局的版本，与 Fire 的项目版本独立演进。
_Avoid_: 项目版本、模型版本

**Model Profile**:
一个被明确识别和验证的具体 Source Checkpoint 及其模型语义约束，而不是对整个模型家族的承诺。
_Avoid_: 模型家族、架构名

**Export Compatibility**:
Source Checkpoint 可被转换为 Fire Tensor Container，且 C++ 能解析全部目录项并核对选定 tensor 的数值。
_Avoid_: 模型支持、可运行

**Model Support**:
Fire 能使用对应 Model Profile 完成端到端生成，并通过约定的参考实现校验。
_Avoid_: 可导出、可读取、格式兼容

**FireReader**:
认识 Fire Tensor Container、按名称提供 tensor 记录和共享数据视图的读取模块。
_Avoid_: ModelLoader、模型解析器

**ModelLoader**:
认识具体 Model Profile、取得所需具名 tensor 并将其绑定到 Fire 模型结构的加载模块。
_Avoid_: FireReader、格式解析器

**Fire Quantization**:
将未量化的 Source Checkpoint 离线转换为 Fire 量化表示的过程，不包括对已量化 checkpoint 的再量化。
_Avoid_: AWQ 导入、量化推理

**Weight-only Quantization**:
仅以低比特表示模型权重，激活值保持原有浮点表示的量化边界。
_Avoid_: 全量化、激活量化

**Canonical Quantized Layout**:
Fire 量化执行路径统一使用的权重、scale 和 zero-point 表示；外部量化格式必须先转换到该表示。
_Avoid_: Source Checkpoint 布局、AWQ 原生布局

**AWQ Compatibility**:
官方 AWQ Source Checkpoint 能被导入为 Canonical Quantized Layout，并达到 Model Support；不表示 Runtime 直接执行 AWQ 原生布局。
_Avoid_: Fire Quantization、AWQ 原生执行
