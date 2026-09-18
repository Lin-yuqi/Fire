# Fire 仓库地图

> 本文描述仓库当前实际状态，用于快速定位代码、理解依赖关系和继续开发。最后核对日期：2026-09-18。

## 1. 项目定位与当前阶段

Fire 是一个以学习和实验为目标、从零构建的轻量级 CUDA 推理框架，计划沿着“内存与 Buffer → Tensor → Operator → Model → Runtime → LLM 推理”的方向演进。

当前仓库已经完成第一版基础抽象，并打通 `.fire` 模型加载到 TinyLlama logits 的 CPU/CUDA 核心链路，处于 v0.1 发布收尾阶段：

- `base` 模块已经接入构建，提供 CPU/GPU 分配器、内存拷贝以及 Buffer 生命周期管理。
- `tensor` 已接入 `Fire::fire`，具备接口和初步实现，并开始覆盖 Buffer 字节偏移与 clone 行为。
- `op` 已建立 `Operator`、`ParamOperator` 与执行上下文 `OpContext`，并接通 Add、RMSNorm、Matmul、Linear、Embedding、RoPE、SwiGLU 和 MHA。Softmax 已有 CPU/CUDA kernel 与直接数值测试，但尚无独立公开 Operator 包装；MHA 的 CPU/CUDA 实现内置稳定 softmax。`ParamOperator` 显式允许移动、禁止复制，支持按值组织 Block 中的参数算子。
- Add 已提供 CPU（Armadillo）与 CUDA FP32 kernel，通过设备类型分派，并覆盖公共 `forward`、参数校验、边界尺寸和非默认 stream 测试。
- RMSNorm 已提供 CPU 一维及 CUDA 一维/二维 FP32 kernel，支持自定义 epsilon 和 CUDA stream；非 4 整数倍行宽会在未对齐行使用标量路径，避免 `float4` 未对齐访问。
- Matmul/Linear 已具备 FP32 CPU/CUDA kernel 路径与 CPU 数值测试；真实 TinyLlama GPU forward 已覆盖模型 shape，专门的 CUDA shape 矩阵和性能基准仍需补齐。
- Embedding 已提供 CPU/CUDA FP32 查表 kernel、公开参数检查和单/多 token 测试。
- RoPE 已按 HF/TinyLlama 的前后半区配对实现 CPU/CUDA 旋转及紧凑 `[max_seq_len, head_dim / 2]` sin/cos cache，并覆盖非零位置、GQA head 数和非默认 stream 测试。
- SwiGLU 已提供 CPU/CUDA FP32 逐元素实现，检查输入/输出 shape，并验证数值、输入只读性及 CUDA block 边界；Softmax kernel 覆盖一维/二维、原地/非原地、长行和数值稳定性。
- `model` 已实现 `FireReader`：通过 `open + fstat + mmap` 校验 `.fire` v1，建立 name index，并通过共享 `Buffer` 维持 mapping 生命周期。
- `tools` 已实现 model-specific TinyLlama exporter 与最小 FireWriter：验证固定的 201 tensor profile，并按两遍流程写出 FP32 `.fire`。
- 独立 260-byte fixture、Reader 异常矩阵、mapping ownership 和 exporter/writer 失败语义均已接入自动化测试。
- `TinyllamaLoader` 已支持按名字读取单个 CPU mmap Tensor view；`load_weights()` 会校验 201 项 canonical tensor 的数量、名称、FP32 dtype 和 shape，在全部成功后发布结构化权重。
- `Model` 纯接口、固定 profile、结构化权重和 Block 参数结构已建立。`TinyLlamaModel` 已实现参数绑定、Runtime/KVCache 分配、RoPE cache、完整单 token forward、K/V 写入与长度提交，以及 reset。真实模型 CPU/GPU 双 token 测试会检查有限 logits、位置推进、非默认 CUDA stream 和 CPU/GPU 一致性。

因此，当前 Base、Tensor、Operator、已实现的 CPU/CUDA kernel、FireReader 和 Loader 均已编入 `Fire::fire`；Python exporter/writer 作为独立工具运行。RMSNorm 和 Linear 的量化配置已能表示，但当前算子会拒绝量化权重。

现阶段已经形成 `Source Checkpoint → exporter → FireReader/Loader → TinyLlamaModel → CPU/CUDA logits` 的核心闭环。v0.1 发布前重点是补强 Loader/状态失败矩阵、加入 Hugging Face 参考对齐，并连接 Tokenizer、Sampler 与聊天生成入口。模型层契约与实施边界见 [模型层设计](model_design_v0_1.md)。

## 2. 顶层导航

```text
Fire/
├── CMakeLists.txt                 # 项目入口：语言、依赖、src/test 子目录
├── readme.md                      # 项目愿景、构建命令和长期路线图
├── CONTEXT.md                     # 领域术语和版本边界
├── include/Fire/                  # 对外头文件
│   ├── base/
│   │   ├── base.h                 # 公共枚举与 NoCopyable
│   │   ├── alloc.h                # 分配器接口、CPU/GPU 分配器及工厂
│   │   └── buffer.h               # Buffer 所有权与容量封装
│   ├── tensor/
│   │   └── tensor.h               # Tensor 元数据、存储、迁移接口
│   ├── model/
│   │   ├── fire_reader.h          # .fire Reader、TensorInfo 与目录项数查询
│   │   ├── tinyllama_loader.h     # 单 Tensor view 与 201 项结构化加载接口
│   │   ├── model.h                # ModelConfig 与 Model 纯接口
│   │   ├── model_weights.h        # 固定 profile 与结构化权重
│   │   ├── kv_cache.h             # TinyLlama K/V 连续存储与序列长度状态
│   │   └── tinyllama.h            # Block、模型接口与 typed Runtime
│   └── op/
│       ├── operator.h             # Operator、参数与执行上下文
│       ├── add.h                  # VecAddOp 接口
│       ├── embedding.h            # EmbeddingOp 参数绑定与查表接口
│       ├── rmsnorm.h              # RmsNormOp 接口与 epsilon 配置
│       ├── matmul.h               # 无绑定参数的 MatmulOp
│       ├── linear.h               # 绑定 weight/可选 bias 的 LinearOp
│       ├── rope.h                 # half-split RoPE 原地旋转接口
│       ├── swiglu.h               # SwiGLU 逐元素算子接口
│       └── mha.h                  # GQA MHA 校验与调度接口
├── src/
│   ├── CMakeLists.txt             # fire 静态库定义；收集模块及 CPU/CUDA kernel
│   ├── base/
│   │   ├── alloc.cpp              # 通用 memcpy/memset 与工厂静态实例
│   │   ├── alloc_cpu.cpp          # malloc/free 后端
│   │   ├── alloc_cu.cpp           # cudaMalloc/cudaFree 后端
│   │   └── buffer.cpp             # Buffer 构造、析构与属性访问
│   ├── tensor/
│   │   └── tensor.cpp             # Tensor 初步实现
│   ├── model/
│   │   ├── fire_reader.cpp        # v1 校验、mmap ownership 与 name index
│   │   ├── tinyllama_loader.cpp   # 201 项 profile 校验与结构化 CPU views
│   │   ├── kv_cache.cpp           # K/V 分配、CPU/GPU 写入、提交与 reset
│   │   └── tinyllama.cpp          # 参数绑定、Runtime 准备与完整 forward
│   ├── op/
│   │   ├── operator.cpp           # Operator 公共检查与参数管理
│   │   ├── add.cpp                # VecAddOp 校验与 kernel 调度
│   │   ├── embedding.cpp          # EmbeddingOp 校验与 kernel 调度
│   │   ├── rmsnorm.cpp            # RmsNormOp 校验与 kernel 调度
│   │   ├── matmul.cpp             # MatmulOp 校验与 kernel 调度
│   │   ├── linear.cpp             # LinearOp 参数检查与 kernel 调度
│   │   ├── rope.cpp               # RoPE shape/cache 校验与 kernel 调度
│   │   ├── swiglu.cpp             # SwiGLU 校验与 kernel 调度
│   │   ├── mha.cpp                # MHA 参数校验与 CPU/CUDA kernel 调度
│   │   └── kernels/
│   │       ├── kernels_interface.*# 按设备类型分派已接入的 CPU/CUDA kernel
│   │       ├── cpu/add_kernel.*   # Armadillo FP32 向量加法
│   │       ├── cpu/emb_kernel.*   # FP32 embedding 行拷贝
│   │       ├── cpu/rmsnorm_kernel.* # CPU FP32 RMSNorm
│   │       ├── cpu/matmul_kernel.* # Armadillo FP32 矩阵乘
│   │       ├── cpu/mha_kernel.*   # CPU FP32 GQA attention
│   │       ├── cpu/rope_kernel.*  # half-split RoPE 与 cache 生成
│   │       ├── cpu/softmax_kernel.* # 稳定 FP32 softmax
│   │       ├── cpu/swiglu_kernel.* # FP32 SwiGLU
│   │       ├── cuda/add_kernel.*  # CUDA FP32 向量加法
│   │       ├── cuda/emb_kernel.*  # CUDA embedding 行拷贝
│   │       ├── cuda/rmsnorm_kernel.* # CUDA FP32 RMSNorm
│   │       ├── cuda/matmul_kernel.* # CUDA FP32 分块矩阵乘
│   │       ├── cuda/mha_kernel.*  # CUDA FP32 GQA attention
│   │       ├── cuda/rope_kernel.* # CUDA RoPE 与 cache 生成
│   │       ├── cuda/softmax_kernel.* # CUDA 稳定 softmax
│   │       └── cuda/swiglu_kernel.* # CUDA FP32 SwiGLU
├── test/
│   ├── CMakeLists.txt             # 单一 fire_tests 测试可执行文件
│   ├── test_base/test_buffer.cpp  # Buffer 自有/外部内存测试
│   ├── test_tensor/test_tensor.cpp# Tensor 字节偏移与 clone 测试
│   ├── test_model/
│   │   ├── test_fire_reader.cpp   # Reader wire/error/lifetime 合同测试
│   │   ├── test_model.cpp         # Block 移动语义与 TinyLlama Norm epsilon
│   │   ├── test_tinyllama_loader.cpp # 真实权重加载与 CPU/GPU forward 集成测试
│   │   ├── test_export_tinyllama.py # Writer/exporter Python 合同测试
│   │   └── generate_fire_v1_fixture.py # 独立 260-byte v1 fixture 生成器
│   ├── test_op/test_op.cpp        # Operator 与 Add 测试
│   ├── test_op/test_emb.cpp       # Embedding 校验、CPU/CUDA 数值测试
│   ├── test_op/test_rmsnorm.cpp   # RMSNorm CPU/CUDA 与错误分支测试
│   ├── test_op/test_matmul.cpp    # CPU Matmul/Linear 数值测试
│   ├── test_op/test_mha.cpp       # MHA CPU/CUDA 数值与错误分支测试
│   ├── test_op/test_rope.cpp      # compact cache、half-split 与 GQA 测试
│   ├── test_op/test_softmax.cpp   # CPU/CUDA 数值稳定性和宽度边界
│   ├── test_op/test_swiglu.cpp    # SwiGLU 数值、校验及输入只读性测试
│   └── utils.cu/.cuh              # CUDA 测试辅助函数
├── docs/
│   ├── model_export_v0_1.md       # TinyLlama 导出、.fire v1 与验收设计
│   ├── model_design_v0_1.md       # Model 契约、权重与执行设计及实施边界
│   └── repo_map.md                # 本文
├── tools/
│   ├── fire_writer.py             # model-agnostic .fire v1 writer
│   └── export_tinyllama.py        # TinyLlama-specific 两遍 exporter/CLI
├── demo/
│   ├── CMakeLists.txt             # llama_chat 目标接入中
│   └── llama_chat.cpp             # v0.1 聊天 CLI 接入中
└── build/                         # 本地生成物，不属于源码
```

## 3. 构建与依赖

顶层 `CMakeLists.txt` 声明 C++17 和 CUDA 17，并要求以下依赖：

| 依赖 | 当前用途 |
| --- | --- |
| CUDA Toolkit | CUDA runtime、GPU 内存分配/拷贝；项目配置阶段即强制需要 |
| glog | `CHECK`、`LOG` 断言与日志 |
| Armadillo | CPU Add/Matmul/SwiGLU kernel 的运算后端 |
| GoogleTest | 仅在 `BUILD_TESTING=ON` 时查找，用于测试 |
| Python 3 | `BUILD_TESTING=ON` 时生成独立 fixture 并运行 Writer/exporter 合同测试；也用于 exporter CLI |
| NumPy、safetensors | Writer/exporter 合同测试与实际导出的 Python 运行时依赖 |
| PyTorch | 实际 TinyLlama payload pass 的 BF16 读取与 FP32 转换；轻量合同测试不加载真实权重 |

构建目标：

```text
Fire (project)
├── fire / Fire::fire             # 静态库；包含 base、tensor、op、model 与 CPU/CUDA kernel
├── fire_v1_fixture               # build-tree 内生成独立 260-byte fixture
├── llama_chat                    # demo 目标，源码与链接配置仍在接入
└── fire_tests                    # 依赖 fixture，链接 Fire::fire + GTest::gtest_main
    ├── test_base/test_buffer.cpp
    ├── test_tensor/test_tensor.cpp
    ├── test_model/test_fire_reader.cpp
    ├── test_model/test_model.cpp
    ├── test_model/test_tinyllama_loader.cpp
    ├── test_op/test_op.cpp
    ├── test_op/test_emb.cpp
    ├── test_op/test_rmsnorm.cpp
    ├── test_op/test_matmul.cpp
    ├── test_op/test_mha.cpp
    ├── test_op/test_rope.cpp
    ├── test_op/test_softmax.cpp
    ├── test_op/test_swiglu.cpp
    └── utils.cu

CTest registration
└── fire_export_tinyllama_contract # Python Writer/exporter unittest
```

常用命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target fire fire_tests -j
ctest --test-dir build --output-on-failure
```

`llama_chat` 尚在接入，因此上面的命令只构建当前已验证的 core/test 目标。CTest 注册 C++ GTest 与 Python Writer/exporter 合同测试；具体通过与跳过数量以当前运行结果为准。真实 TinyLlama `.fire` 已按 201 项、`data_offset = 19,328`、文件长度 `4,400,212,864` bytes 完成加载和 CPU/GPU forward；从 source checkpoint 抽样做 bit-exact 数值回读仍未形成独立验收记录。

## 4. 模块关系

```text
base::DeviceType / DataType / MemCpyKind
                    │
                    ▼
          base::DeviceAllocator
             ├── CPUAllocator ── malloc/free
             └── GPUAllocator ── cudaMalloc/cudaFree
                    │
                    ▼
              base::Buffer
       （指针、容量、设备、所有权）
                    │
                    ▼
              tensor::Tensor
       （dtype、shape、stride、offset）

op::VecAddOp ── 校验 Tensor 与执行上下文
      │
      ▼
kernel::get_add_kernel(DeviceType)
      ├── CPU ── Armadillo 向量加法
      └── GPU ── CUDA FP32 kernel（支持可选 stream）

op::RmsNormOp ── 校验输入、输出与 weight
      │
      ▼
kernel::get_rmsnorm_kernel[_dim](DeviceType)
      ├── CPU ── 一维 FP32 RMSNorm
      └── GPU ── 一维/二维 FP32 RMSNorm（支持可选 stream）

EmbeddingOp / RoPEOp / SwiGLUOp
      │  校验 dtype、shape、device 与算子特有约束
      ▼
kernel::get_*_kernel(DeviceType)
      ├── CPU ── FP32 reference/Armadillo 路径
      └── GPU ── CUDA FP32 kernel（支持调用方 stream）

Softmax（当前仅 kernel 接口）
      ├── CPU ── 稳定逐行 softmax
      └── GPU ── block reduction；长行使用通用循环路径

MultiHeadAttentionOp
      │  校验 Q、完整 K/V cache、score/output、layer 与 pos
      ▼
kernel::get_mha_kernel(DeviceType)
      ├── CPU ── FP32 GQA attention
      └── GPU ── FP32 GQA attention（支持调用方 stream）

TinyLlama source checkpoint
      │
      ▼
export_tinyllama.py ── FireWriter ── .fire v1 ── FireReader
                                                    │
                                                    ▼
                                      shared mmap Buffer + Tensor view
```

### `base`: 设备、分配器与 Buffer

命名空间为 `base`。

- `base.h` 定义设备类型 `CPU/GPU/Unknown`、拷贝方向、数据类型以及禁止复制的基类。
- `DeviceAllocator` 是分配后端抽象，统一提供 `allocate`、`release`、`memcpy`、`memset_zero`。
- `CPUAllocator` 使用 `malloc/free`；`GPUAllocator` 使用 CUDA runtime。
- 两个 Factory 返回进程级共享分配器实例。
- `Buffer` 有两种模式：通过 allocator 分配并拥有内存，或包装外部指针且不负责释放。它是不可复制对象，可通过 `shared_ptr` 共享底层存储；`get()` 基于 `enable_shared_from_this` 返回当前 Buffer 的共享引用。

Buffer 已接入库目标和测试目标，也是 Tensor 底层存储的基础。

### `tensor`: 形状与存储视图

命名空间为 `tensor`。`Tensor` 在 `Buffer` 上提供：

- 数据类型、维度、元素数量和连续行主序 stride；
- 仅元数据、allocator 分配、复用 Buffer、包装外部指针等构造路径；
- reshape、clone、CPU/CUDA 迁移和存储重新绑定；
- 以字节为单位的 Buffer offset；`raw_ptr()` 返回应用该偏移后的地址；
- 对外通过 `ptr<T>()` 提供类型化地址，应用 offset 的 `raw_ptr()` 仅供 Tensor 内部使用；
- clone 从偏移后的有效数据起点复制，迁移到新 Buffer 后 offset 归零；
- `to_cuda(stream)` 是同步迁移接口：非空 stream 的异步拷贝会在返回前完成并检查同步结果，之后才释放原 CPU Buffer。

Tensor 已编入 `fire`，目前有 `from_blob`、字节偏移和 CPU clone 测试；设备迁移与更多行为测试仍待完善，详见第 6 节。

### `model`: Reader、Loader 与 TinyLlama Runtime

命名空间为 `model`。`TensorInfo` 是 FireReader 对调用方返回的 name、dtype、dims、绝对 byte offset 和 byte size；它不是 96-byte wire struct。`FireReader` 提供以下公共 seam：

- `open(path)` 使用 `open + fstat + mmap` 读取文件，对 Header、Directory、payload 范围、连续性和文件尾执行完整 v1 校验，并返回 `base::Status`。
- `find(name)` 在 Reader 未打开或名称不存在时返回 `nullptr`。
- `tensor_count()` 返回已解析的目录项数，尚未成功打开时为 `0`；Reader 不判断某个模型需要多少项。
- `mapped_buffer()` 返回整个 mmap 的共享 ownership；由此构造的 Tensor view 可以活得比 Reader 更久。

`open()` 先在局部 RAII 状态中完成解析与 name index 构建，成功后才提交成员状态。Reader 已编入 `Fire::fire`；独立 wire fixture、坏文件矩阵、`uint64` 溢出、Status 分类和 mapping lifetime 测试均为 active。

`TinyllamaLoader::open()` 复用 Reader 的容器校验；`loader_tensor(name, output)` 根据 metadata 构造共享 mmap 的 CPU Tensor view，不验证 TinyLlama profile。`load_weights(output)` 要求目录数量为 201，按 canonical name 逐项校验 FP32 dtype 和固定 shape，在局部对象中完成全部 view 组装后才移动发布 `TinyLlamaWeights`；任一失败不会发布部分结果。当前真实 `.fire` 集成测试会在文件存在时验证结构和 Loader 销毁后的 mmap ownership，但系统性的缺项、额外项和错误 shape 测试仍待补齐。

`model.h` 定义单序列 `Model` 的 `config/prepare/forward/reset` 纯接口；`model_weights.h` 定义唯一的 C++ 固定 profile 与按层组织的 Tensor 字段。`TinyLlamaModel::create` 把结构化权重绑定到 Embedding、Norm 和 Linear 参数算子，并按 context 决定是否迁移到 GPU；`prepare` 分配 typed Runtime、K/V Tensor、attention score 和紧凑 RoPE cache。`forward` 已按 22 层顺序执行 Attention/FFN、写入 K/V、生成 logits，并在全部成功后提交 cache length；`reset` 将有效长度归零并保留已分配存储。

### `tools`: exporter/writer

`fire_writer.py` 固定 `.fire` v1 常量、wire dtype 和规范化 `TensorInfo`，在写入前校验 name、rank、shape、byte size 及紧密排列的 absolute offset。`FireWriter` 编码 32-byte Header 和 96-byte Directory entry，再通过 `memoryview(tensor).cast("B")` 写入单个 FP32 C-contiguous NumPy payload。

`export_tinyllama.py` 固定 TinyLlama config、BF16 source dtype、canonical name/shape profile、201 个 tensor 和预期文件大小。它先用 NumPy safetensors metadata pass 完成 preflight，再 exclusive-create 目标，通过单个长期 PyTorch `safe_open` context 逐 tensor 转为 FP32 并写出；已创建目标后遇到普通异常或 `KeyboardInterrupt` 会关闭并 best-effort 删除 partial file。

独立 fixture 生成器与 Python 合同测试已接入 CTest；`fire_v1_fixture` 是 `fire_tests` 的构建依赖，而非需要手动运行的非默认前置。本阶段没有引入通用 model adapter、registry、provider、planner 或 streaming framework。

### `op`: Operator 与 FP32 算子

命名空间为 `op`。`Operator` 保存算子类型和名称，并提供 Tensor 非空、设备、数据类型及维度检查；`ParamOperator` 管理带量化配置的参数列表，显式禁止复制并支持移动，使参数算子可按值放入 Block 容器。`OpContext` 描述单次执行的设备、CUDA stream、allocator 和 workspace。

`VecAddOp::forward` 先检查三个 Tensor 的设备、数据类型与 shape，再根据 `OpContext::_device_type` 选择 CPU 或 CUDA kernel。当前 Add 只实现 FP32：CPU 路径通过 Armadillo `fvec` 相加，CUDA 路径由每个线程处理一个元素并进行边界检查，也可使用调用方传入的 stream。

`RmsNormOp::forward` 从输入最后一维确定归一化宽度，检查 FP32 输入、输出和单个未量化 weight 参数，并按设备和输入维数选择 kernel。CPU 路径当前只接受一维输入；GPU 路径覆盖一维向量和二维 `{rows, width}` 输入。二维 kernel 在行起点满足 16 字节对齐时使用 `float4`，否则回退到标量访问，因此 width 不是 4 的整数倍时不会对后续行进行未对齐的 `float4` 访问。

Add 测试已覆盖公共 `VecAddOp::forward` 的 CPU/GPU 路径、参数错误、边界尺寸和非默认 stream。RMSNorm 测试覆盖 CPU 正确性、自定义 epsilon、参数错误、CUDA 打包尾部以及二维非 4 整数倍行宽。

`MatmulOp` 接受一维或二维输入和 `[out_features, in_features]` 权重，计算 `scale * input * weight^T`。CPU 使用 Armadillo，CUDA 使用分块 kernel。`LinearOp` 绑定 weight 和可选 bias，复用 Matmul/Add kernel，并拒绝量化参数。现有 Matmul/Linear 数值测试覆盖 CPU 向量/矩阵及向量 bias；真实 TinyLlama GPU forward 已覆盖模型使用的 CUDA shape，专门的 CUDA 数值矩阵和性能基准仍需补齐。

`EmbeddingOp` 绑定一个未量化 FP32 `[vocab_size, embedding_dim]` weight，接收同设备 INT32 token Tensor，并要求输出元素数等于 token 数乘 embedding width。CPU/CUDA kernel 都按 token 顺序复制权重行；测试覆盖单/多 token、错误参数和 CUDA 非对齐尾部。

`RoPEOp` 原地修改二维 `[heads, head_dim]` query/key。`head_dim` 必须为正偶数，key heads 不得多于 query heads；sin/cos cache 使用 `[max_seq_len, head_dim / 2]`，每个 head 内按前后半区配对，符合未 permutation 的 HF Q/K 权重布局。CPU/CUDA cache 生成与旋转测试覆盖非零位置、GQA 和非默认 stream。

`SwiGLUOp` 对同 shape FP32 Tensor 计算 `SiLU(gate) * up`。CPU/CUDA 实现都写入独立输出并保持 const 输入不变；测试覆盖一维/二维、错误 Tensor、selector 和 CUDA block 边界。

Softmax 当前通过 `get_softmax_kernel` 和 CPU/CUDA kernel 暴露，支持一维向量或二维逐行计算、原地或非原地输出；CUDA 对宽度不超过 2048 的行使用固定寄存器项数，对更长行使用通用循环。它已有宽度边界、极值、等值 logits 和 `-inf` mask 测试，但尚无 `SoftmaxOp` 公共包装。

`MultiHeadAttentionOp` 接收 `[num_q_heads, head_dim]` query、四维连续 K/V cache、`[num_q_heads, capacity]` score buffer，以及 `layer_idx/pos`。它校验 GQA head 整除关系和所有 shape，再按设备分派 CPU/CUDA kernel；kernel 只计算 `[0, pos]` 的有效前缀，并将 softmax 概率与对应 value 累积为 `[num_q_heads, head_dim]` 输出。测试覆盖 CPU 参考值、输入只读性、参数错误和可用时的 CUDA 路径。

## 5. 关键运行路径

Add 的当前调用路径是：

1. 调用方准备设备、dtype 和 shape 一致的输入与输出 Tensor，并填写 `OpContext`。
2. `VecAddOp::forward` 执行公共 Tensor 检查和 shape 检查。
3. `get_add_kernel` 按 CPU/GPU 返回对应函数指针。
4. CPU kernel 使用 Armadillo 写入输出；CUDA kernel 按元素写入输出，并使用可选 stream 启动。

RMSNorm 的当前调用路径是：

1. 调用方设置一个与最后一维等长的 FP32 weight，并准备与输入 shape 相同的输出 Tensor。
2. `RmsNormOp::forward` 校验设备、dtype、shape、参数数量和量化状态。
3. 一维输入通过 `get_rmsnorm_kernel` 分派；二维 GPU 输入通过 `get_rmsnorm_kernel_dim` 分派，每个 block 处理一行。
4. CUDA kernel 对可安全打包的地址使用 `float4`，并用标量循环处理尾部或未对齐行。

`.fire` 权重的当前路径是：

1. `export_tinyllama.py` 在目标创建前校验 config 和 201 项 source metadata。
2. `FireWriter` 按规划好的绝对 offset 写入 Header、Directory 和逐项 FP32 payload。
3. `FireReader::open` mmap 整个文件并验证所有 v1 不变量，`find` 按 canonical name 返回解码后 metadata。
4. `TinyllamaLoader::load_weights` 按 canonical name 校验 201 项 dtype/shape，用 `mapped_buffer()` 与 absolute byte offset 组装结构化 CPU Tensor views。
5. `TinyLlamaModel::create/prepare` 绑定参数并准备 Runtime；`forward` 依次执行 Embedding、22 层 Attention/FFN、final RMSNorm 和 LM Head，得到 `[32000]` logits。
6. 每层 K/V 写入 `[layer, pos, ...]`，MHA 读取 `[0, pos]` 的有效历史；整个 token 成功后 `KVCache::commit(pos)` 才推进长度。

外部内存路径则由 `Buffer(ptr, capacity, device_type)` 包装；`owns_memory()` 为 false，调用方仍负责外部指针的生命周期。

Operator 的可恢复参数错误通过 `base::Status` 返回；kernel 内部约束和未知设备仍使用 glog `CHECK/LOG(FATAL)`。CUDA 运行失败仍应检查对应 API 或 kernel launch 的错误码。

## 6. 当前边界与已知技术债

以下是阅读和继续开发时最重要的事实，不等同于本次要修复的任务清单：

1. **部分 CUDA 返回值未检查**：部分 memcpy/memset 路径忽略 CUDA API 返回的 `cudaError_t`，失败时可能缺少及时、准确的错误信息；按当前约定可继续使用 `CHECK/LOG(FATAL)` 报错，无需引入额外错误类型。
2. **RMSNorm 支持范围有限**：当前只有 FP32 实现，CPU 仅支持一维输入，GPU 多行路径已验证二维 `{rows, width}`；更高维输入的所有前导维度尚未完整接入 block 调度。量化 weight 会返回 `InvalidArgument`。
3. **模型核心闭环已完成，上层生成仍缺失**：真实约 4.1 GiB `.fire` 已通过 201 项 Loader 校验和 CPU/GPU 双 token forward；但尚未接入 tokenizer、sampler、停止条件和可用的聊天 CLI，也尚未完成 Hugging Face 参考 logits 对齐，因此 v0.1 仍处于发布收尾阶段。
4. **Matmul 验证与量化仍待补齐**：已有 FP32 CPU/CUDA 实现、CPU 单算子数值测试和真实模型 GPU 路径；更系统的 CUDA shape/误差矩阵、性能优化和量化路径属于后续工作。
5. **Softmax 尚未形成独立 Operator**：CPU/CUDA kernel 和直接测试已完成，MHA 已在内部封装 softmax 计算；其他调用方目前仍需要直接使用 kernel 接口。
6. **状态错误矩阵仍不完整**：KVCache 小尺寸测试已覆盖 K/V 写入、提交、reset、错误 shape 与错误位置，真实模型 forward 也会拒绝重复位置；更多 create/prepare/reset 失败保持场景仍需补齐。

## 7. 修改入口速查

| 想做的事情 | 首要入口 | 通常还需同步 |
| --- | --- | --- |
| 增加数据类型/设备类型 | `include/Fire/base/base.h` | dtype 字节数、拷贝/分配逻辑、测试 |
| 修改 CPU/GPU 分配策略 | `include/Fire/base/alloc.h`、`src/base/alloc_*.cpp` | Buffer 行为与 allocator 测试 |
| 修改存储所有权 | `include/Fire/base/buffer.h`、`src/base/buffer.cpp` | Buffer/Tensor 测试 |
| 完善 Tensor | `include/Fire/tensor/tensor.h`、`src/tensor/tensor.cpp` | `src/CMakeLists.txt`、`test/test_tensor/` |
| 增加 Operator | `include/Fire/op/`、`src/op/` | `src/op/kernels/`、`src/CMakeLists.txt`、对应测试目录 |
| 修改 RMSNorm | `include/Fire/op/rmsnorm.h`、`src/op/rmsnorm.cpp` | CPU/CUDA kernel、接口分派、`test/test_op/test_rmsnorm.cpp` |
| 修改 `.fire` 读写 | `tools/fire_writer.py`、`include/Fire/model/fire_reader.h` | `src/model/fire_reader.cpp`、`test/test_model/`、格式设计文档 |
| 增加模型或 Runtime | `include/Fire/model/` 与 `src/model/` | CMake 源文件、测试、README/本文 |
| 增加测试 | `test/test_<module>/` | `test/CMakeLists.txt` |
| 增加 Python 导出工具 | `tools/` | 对应 Python test；无需为纯脚本增加 tools CMake |
| 增加编译型 benchmark 工具 | `tools/` | 顶层 `add_subdirectory(tools)` 与 tools CMake |

新增实现文件时务必显式确认它已进入某个 CMake target；“文件存在”并不意味着会被编译。

## 8. 近期开发计划

当前模型核心链路已形成闭环，v0.1 发布前按以下顺序推进：

1. 为真实导出补充 source checkpoint 抽样 bit-exact FP32 回读，并完成 Loader 的缺项、额外项和错误 shape 失败矩阵。
2. 用固定 token 序列与 Hugging Face 参考实现逐位置比较 logits，建立独立于 CPU/GPU 自比较的数值基线。
3. 接入 Tokenizer、Sampler、停止条件和 `llama_chat`，完成可运行的自回归生成示例。
4. 整理 v0.1 release 配置、命令、已知限制与性能基线。
5. v0.1 后继续推进 Matmul 量化、Kernel Fusion、显存复用和 profiling。

## 9. 文档维护约定

当出现以下变化时应同步更新本文件：新增顶层模块、CMake target 或外部依赖；模块从“占位”变为“接入构建”；关键数据流或所有权规则改变；已知边界被修复或替换。
