# Fire 仓库地图

> 本文描述仓库当前实际状态，用于快速定位代码、理解依赖关系和继续开发。最后核对日期：2026-09-11。

## 1. 项目定位与当前阶段

Fire 是一个以学习和实验为目标、从零构建的轻量级 CUDA 推理框架，计划沿着“内存与 Buffer → Tensor → Operator → Model → Runtime → LLM 推理”的方向演进。

当前仓库已经完成第一版基础抽象，并打通 Fire Tensor Container 的写入、读取和自动化合同测试：

- `base` 模块已经接入构建，提供 CPU/GPU 分配器、内存拷贝以及 Buffer 生命周期管理。
- `tensor` 已接入 `Fire::fire`，具备接口和初步实现，并开始覆盖 Buffer 字节偏移与 clone 行为。
- `op` 已建立 `Operator`、`ParamOperator` 与执行上下文 `OpContext`，并接通向量 Add 和 RMSNorm。
- Add 已提供 CPU（Armadillo）与 CUDA FP32 kernel，通过设备类型分派，并覆盖公共 `forward`、参数校验、边界尺寸和非默认 stream 测试。
- RMSNorm 已提供 CPU 一维及 CUDA 一维/二维 FP32 kernel，支持自定义 epsilon 和 CUDA stream；非 4 整数倍行宽会在未对齐行使用标量路径，避免 `float4` 未对齐访问。
- `model` 已实现 `FireReader`：通过 `open + fstat + mmap` 校验 `.fire` v1，建立 name index，并通过共享 `Buffer` 维持 mapping 生命周期。
- `tools` 已实现 model-specific TinyLlama exporter 与最小 FireWriter：验证固定的 201 tensor profile，并按两遍流程写出 FP32 `.fire`。
- 独立 260-byte fixture、Reader 异常矩阵、mapping ownership 和 exporter/writer 失败语义均已接入自动化测试。
- runtime、TinyLlama ModelLoader 和模型执行结构仍未建立。

因此，当前 Base、Tensor、Operator、Add、RMSNorm kernel 和 FireReader 均已编入 `Fire::fire`；Python exporter/writer 作为独立工具运行。RMSNorm 的量化配置已能表示，但当前算子会拒绝量化权重。

现阶段将 `DeviceAllocator → Buffer → Tensor → Operator → kernel 分派` 和 `Source Checkpoint → exporter → FireWriter → FireReader` 视为可继续扩展的基础结构。下一步先完成真实 TinyLlama 全量导出与回读里程碑，再进入 ModelLoader 和模型执行；实际开发中发现接口缺口时再做针对性调整。

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
│   │   └── fire_reader.h          # .fire Reader 与解码后 TensorInfo 接口
│   └── op/
│       ├── operator.h             # Operator、参数与执行上下文
│       ├── add.h                  # VecAddOp 接口
│       └── rmsnorm.h              # RmsNormOp 接口与 epsilon 配置
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
│   │   └── fire_reader.cpp        # v1 校验、mmap ownership 与 name index
│   ├── op/
│   │   ├── operator.cpp           # Operator 公共检查与参数管理
│   │   ├── add.cpp                # VecAddOp 校验与 kernel 调度
│   │   ├── rmsnorm.cpp            # RmsNormOp 校验与 kernel 调度
│   │   └── kernels/
│   │       ├── kernels_interface.*# 按设备类型分派 Add/RMSNorm kernel
│   │       ├── cpu/add_kernel.*   # Armadillo FP32 向量加法
│   │       ├── cpu/rmsnorm_kernel.* # CPU FP32 RMSNorm
│   │       ├── cuda/add_kernel.*  # CUDA FP32 向量加法
│   │       └── cuda/rmsnorm_kernel.* # CUDA FP32 RMSNorm
├── test/
│   ├── CMakeLists.txt             # 单一 fire_tests 测试可执行文件
│   ├── test_base/test_buffer.cpp  # Buffer 自有/外部内存测试
│   ├── test_tensor/test_tensor.cpp# Tensor 字节偏移与 clone 测试
│   ├── test_model/
│   │   ├── test_fire_reader.cpp   # Reader wire/error/lifetime 合同测试
│   │   ├── test_export_tinyllama.py # Writer/exporter Python 合同测试
│   │   └── generate_fire_v1_fixture.py # 独立 260-byte v1 fixture 生成器
│   ├── test_op/test_op.cpp        # Operator 与 Add 测试
│   ├── test_op/test_rmsnorm.cpp   # RMSNorm CPU/CUDA 与错误分支测试
│   └── utils.cu/.cuh              # CUDA 测试辅助函数
├── docs/
│   ├── model_export_v0_1.md       # TinyLlama 导出、.fire v1 与验收设计
│   └── repo_map.md                # 本文
├── tools/
│   ├── fire_writer.py             # model-agnostic .fire v1 writer
│   └── export_tinyllama.py        # TinyLlama-specific 两遍 exporter/CLI
└── build/                         # 本地生成物，不属于源码
```

## 3. 构建与依赖

顶层 `CMakeLists.txt` 声明 C++17 和 CUDA 17，并要求以下依赖：

| 依赖 | 当前用途 |
| --- | --- |
| CUDA Toolkit | CUDA runtime、GPU 内存分配/拷贝；项目配置阶段即强制需要 |
| glog | `CHECK`、`LOG` 断言与日志 |
| Armadillo | CPU Add kernel 的向量运算后端 |
| GoogleTest | 仅在 `BUILD_TESTING=ON` 时查找，用于测试 |
| Python 3 | `BUILD_TESTING=ON` 时生成独立 fixture 并运行 Writer/exporter 合同测试；也用于 exporter CLI |
| NumPy、safetensors | Writer/exporter 合同测试与实际导出的 Python 运行时依赖 |
| PyTorch | 实际 TinyLlama payload pass 的 BF16 读取与 FP32 转换；轻量合同测试不加载真实权重 |

构建目标：

```text
Fire (project)
├── fire / Fire::fire             # 静态库；包含 base、tensor、op、FireReader 与 CPU/CUDA kernel
├── fire_v1_fixture               # build-tree 内生成独立 260-byte fixture
└── fire_tests                    # 依赖 fixture，链接 Fire::fire + GTest::gtest_main
    ├── test_base/test_buffer.cpp
    ├── test_tensor/test_tensor.cpp
    ├── test_model/test_fire_reader.cpp
    ├── test_op/test_op.cpp
    ├── test_op/test_rmsnorm.cpp
    └── utils.cu

CTest registration
└── fire_export_tinyllama_contract # Python Writer/exporter unittest
```

常用命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

当前 build 配置的最近验证结果为 CTest 26 项、0 失败、4 项因现有 CUDA/环境条件跳过；Python Writer/exporter 合同的 13 个 unittest 均通过。真实 TinyLlama metadata 已确认为 201 项、`data_offset = 19,328`、计划文件长度 `4,400,212,864` bytes，但未在本阶段执行完整 4.4 GB payload 导出与回读。

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

### `model`: FireReader

命名空间为 `model`。`TensorInfo` 是 FireReader 对调用方返回的 name、dtype、dims、绝对 byte offset 和 byte size；它不是 96-byte wire struct。`FireReader` 提供以下公共 seam：

- `open(path)` 使用 `open + fstat + mmap` 读取文件，对 Header、Directory、payload 范围、连续性和文件尾执行完整 v1 校验，并返回 `base::Status`。
- `find(name)` 在 Reader 未打开或名称不存在时返回 `nullptr`。
- `mapped_buffer()` 返回整个 mmap 的共享 ownership；由此构造的 Tensor view 可以活得比 Reader 更久。

`open()` 先在局部 RAII 状态中完成解析与 name index 构建，成功后才提交成员状态。Reader 已编入 `Fire::fire`；独立 wire fixture、坏文件矩阵、`uint64` 溢出、Status 分类和 mapping lifetime 测试均为 active。

### `tools`: exporter/writer

`fire_writer.py` 固定 `.fire` v1 常量、wire dtype 和规范化 `TensorInfo`，在写入前校验 name、rank、shape、byte size 及紧密排列的 absolute offset。`FireWriter` 编码 32-byte Header 和 96-byte Directory entry，再通过 `memoryview(tensor).cast("B")` 写入单个 FP32 C-contiguous NumPy payload。

`export_tinyllama.py` 固定 TinyLlama config、BF16 source dtype、canonical name/shape profile、201 个 tensor 和预期文件大小。它先用 NumPy safetensors metadata pass 完成 preflight，再 exclusive-create 目标，通过单个长期 PyTorch `safe_open` context 逐 tensor 转为 FP32 并写出；已创建目标后遇到普通异常或 `KeyboardInterrupt` 会关闭并 best-effort 删除 partial file。

独立 fixture 生成器与 Python 合同测试已接入 CTest；`fire_v1_fixture` 是 `fire_tests` 的构建依赖，而非需要手动运行的非默认前置。本阶段没有引入通用 model adapter、registry、provider、planner 或 streaming framework。

### `op`: Operator、Add 与 RMSNorm

命名空间为 `op`。`Operator` 保存算子类型和名称，并提供 Tensor 非空、设备、数据类型及维度检查；`ParamOperator` 管理带量化配置的参数列表。`OpContext` 描述单次执行的设备、CUDA stream、allocator 和 workspace。

`VecAddOp::forward` 先检查三个 Tensor 的设备、数据类型与 shape，再根据 `OpContext::_device_type` 选择 CPU 或 CUDA kernel。当前 Add 只实现 FP32：CPU 路径通过 Armadillo `fvec` 相加，CUDA 路径由每个线程处理一个元素并进行边界检查，也可使用调用方传入的 stream。

`RmsNormOp::forward` 从输入最后一维确定归一化宽度，检查 FP32 输入、输出和单个未量化 weight 参数，并按设备和输入维数选择 kernel。CPU 路径当前只接受一维输入；GPU 路径覆盖一维向量和二维 `{rows, width}` 输入。二维 kernel 在行起点满足 16 字节对齐时使用 `float4`，否则回退到标量访问，因此 width 不是 4 的整数倍时不会对后续行进行未对齐的 `float4` 访问。

Add 测试已覆盖公共 `VecAddOp::forward` 的 CPU/GPU 路径、参数错误、边界尺寸和非默认 stream。RMSNorm 测试覆盖 CPU 正确性、自定义 epsilon、参数错误、CUDA 打包尾部以及二维非 4 整数倍行宽。

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
4. 调用方用 `mapped_buffer()` 与 absolute byte offset 构造 Tensor view；TinyLlama ModelLoader 尚未实现。

外部内存路径则由 `Buffer(ptr, capacity, device_type)` 包装；`owns_memory()` 为 false，调用方仍负责外部指针的生命周期。

Operator 的可恢复参数错误通过 `base::Status` 返回；kernel 内部约束和未知设备仍使用 glog `CHECK/LOG(FATAL)`。CUDA 运行失败仍应检查对应 API 或 kernel launch 的错误码。

## 6. 当前边界与已知技术债

以下是阅读和继续开发时最重要的事实，不等同于本次要修复的任务清单：

1. **部分 CUDA 返回值未检查**：部分 memcpy/memset 路径忽略 CUDA API 返回的 `cudaError_t`，失败时可能缺少及时、准确的错误信息；按当前约定可继续使用 `CHECK/LOG(FATAL)` 报错，无需引入额外错误类型。
2. **RMSNorm 支持范围有限**：当前只有 FP32 实现，CPU 仅支持一维输入，GPU 多行路径已验证二维 `{rows, width}`；更高维输入的所有前导维度尚未完整接入 block 调度。量化 weight 会返回 `InvalidArgument`。
3. **模型导入尚未完整闭环**：`.fire` v1 写入、TinyLlama-specific 两遍 exporter、mmap Reader 及自动化 wire/profile 合同测试已完成；完整 4.4 GB 真实导出与选定数值回读、TinyLlama ModelLoader、runtime、MatMul 及其量化 kernel 仍未完成。因此尚不宣称 Export Compatibility 或 Model Support 达标。

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

当前基础抽象已形成初步闭环，近期按以下顺序推进：

1. 执行真实 TinyLlama 全量导出，用 FireReader 解析 201 项并对选定元素做 bit-exact FP32 回读，完成 Export Compatibility 验收。
2. 实现 TinyLlama ModelLoader，再接入模型 Layer 和端到端生成校验。
3. 在真实模型 shape 上建立 FP32 MatMul 基线及正确性测试，明确 Operator、weight 布局和 kernel 分派接口。
4. 在 FP32 基线稳定后准备 MatMul 量化路径；复用现有 `QuantConfig` 表达量化方式，并分别验证量化参数、数值误差和 CUDA kernel 边界。
5. 后续再连接 Transformer 其他算子、KV cache、Tokenizer 与自回归 Runtime。

## 9. 文档维护约定

当出现以下变化时应同步更新本文件：新增顶层模块、CMake target 或外部依赖；模块从“占位”变为“接入构建”；关键数据流或所有权规则改变；已知边界被修复或替换。
