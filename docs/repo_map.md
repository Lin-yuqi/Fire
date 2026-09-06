# Fire 仓库地图

> 本文描述仓库当前实际状态，用于快速定位代码、理解依赖关系和继续开发。最后核对日期：2026-09-06。

## 1. 项目定位与当前阶段

Fire 是一个以学习和实验为目标、从零构建的轻量级 CUDA 推理框架，计划沿着“内存与 Buffer → Tensor → Operator → Model → Runtime → LLM 推理”的方向演进。

当前仓库已经完成第一版基础抽象，开发重点正从搭建框架转向补齐测试和扩展算子：

- `base` 模块已经接入构建，提供 CPU/GPU 分配器、内存拷贝以及 Buffer 生命周期管理。
- `tensor` 已接入 `Fire::fire`，具备接口和初步实现，并开始覆盖 Buffer 字节偏移与 clone 行为。
- `op` 已建立 `Operator`、`ParamOperator` 与执行上下文 `OpContext`，并初步接通向量 Add。
- Add 已提供 CPU（Armadillo）与 CUDA FP32 kernel，通过设备类型分派；GPU 测试已加入测试目标。
- `model`、runtime、工具程序等仍是规划或空目录，尚无实现。

因此，当前 Base、Tensor、Operator 和 Add kernel 均已编入 `Fire::fire`；现有 Add 测试直接覆盖 CUDA kernel 路径，尚未覆盖 `VecAddOp::forward` 和 CPU kernel。

现阶段将 `DeviceAllocator → Buffer → Tensor → Operator → kernel 分派` 视为可继续扩展的基础结构。近期不计划优先重构这些抽象，而是先用更完整的测试验证边界，再复用 Add 已建立的组织方式开发其他算子；实际开发中发现接口缺口时再做针对性调整。

## 2. 顶层导航

```text
Fire/
├── CMakeLists.txt                 # 项目入口：语言、依赖、src/test 子目录
├── readme.md                      # 项目愿景、构建命令和长期路线图
├── include/Fire/                  # 对外头文件
│   ├── base/
│   │   ├── base.h                 # 公共枚举与 NoCopyable
│   │   ├── alloc.h                # 分配器接口、CPU/GPU 分配器及工厂
│   │   └── buffer.h               # Buffer 所有权与容量封装
│   ├── tensor/
│   │   └── tensor.h               # Tensor 元数据、存储、迁移接口
│   └── op/
│       ├── operator.h             # Operator、参数与执行上下文
│       └── add.h                  # VecAddOp 接口
├── src/
│   ├── CMakeLists.txt             # fire 静态库定义；收集模块及 CPU/CUDA kernel
│   ├── base/
│   │   ├── alloc.cpp              # 通用 memcpy/memset 与工厂静态实例
│   │   ├── alloc_cpu.cpp          # malloc/free 后端
│   │   ├── alloc_cu.cpp           # cudaMalloc/cudaFree 后端
│   │   └── buffer.cpp             # Buffer 构造、析构与属性访问
│   ├── tensor/
│   │   └── tensor.cpp             # Tensor 初步实现
│   ├── op/
│   │   ├── operator.cpp           # Operator 公共检查与参数管理
│   │   ├── add.cpp                # VecAddOp 校验与 kernel 调度
│   │   └── kernels/
│   │       ├── kernels_interface.*# 按设备类型分派 Add kernel
│   │       ├── cpu/add_kernel.*   # Armadillo FP32 向量加法
│   │       └── cuda/add_kernel.*  # CUDA FP32 向量加法
├── test/
│   ├── CMakeLists.txt             # 单一 fire_tests 测试可执行文件
│   ├── test_base/test_buffer.cpp  # Buffer 自有/外部内存测试
│   ├── test_tensor/test_tensor.cpp# Tensor 字节偏移与 clone 测试
│   ├── test_op/test_op.cpp        # CUDA Add kernel 初步测试
│   └── utils.cu/.cuh              # CUDA 测试辅助函数
├── docs/
│   └── repo_map.md                # 本文
├── tools/                         # 预留空目录
└── build/                         # 本地生成物，不属于源码
```

## 3. 构建与依赖

顶层 `CMakeLists.txt` 声明 C++17 和 CUDA 14，并要求以下依赖：

| 依赖 | 当前用途 |
| --- | --- |
| CUDA Toolkit | CUDA runtime、GPU 内存分配/拷贝；项目配置阶段即强制需要 |
| glog | `CHECK`、`LOG` 断言与日志 |
| Armadillo | CPU Add kernel 的向量运算后端 |
| GoogleTest | 仅在 `BUILD_TESTING=ON` 时查找，用于测试 |

构建目标：

```text
Fire (project)
├── fire / Fire::fire             # 静态库；包含 base、tensor、op 与 CPU/CUDA kernel
└── fire_tests                    # 链接 Fire::fire + GTest::gtest_main
    ├── test_base/test_buffer.cpp
    ├── test_tensor/test_tensor.cpp
    ├── test_op/test_op.cpp
    └── utils.cu
```

常用命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

当前环境中已有 build 配置；本次核对时增量构建成功。由于当前运行环境禁止访问 GPU，CUDA Add 测试未能在本次文档更新中完成运行验证。

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

命名空间为 `tensor`。`Tensor` 计划在 `Buffer` 上增加：

- 数据类型、维度、元素数量和连续行主序 stride；
- 仅元数据、allocator 分配、复用 Buffer、包装外部指针等构造路径；
- reshape、clone、CPU/CUDA 迁移和存储重新绑定；
- 以字节为单位的 Buffer offset；`raw_ptr()` 返回应用该偏移后的地址；
- 对外通过 `ptr<T>()` 提供类型化地址，应用 offset 的 `raw_ptr()` 仅供 Tensor 内部使用；
- clone 从偏移后的有效数据起点复制，迁移到新 Buffer 后 offset 归零；
- `to_cuda(stream)` 是同步迁移接口：非空 stream 的异步拷贝会在返回前完成并检查同步结果，之后才释放原 CPU Buffer。

Tensor 已编入 `fire`，目前有 `from_blob`、字节偏移和 CPU clone 测试；设备迁移与更多行为测试仍待完善，详见第 6 节。

### `op`: Operator 与 Add

命名空间为 `op`。`Operator` 保存算子类型和名称，并提供 Tensor 非空、设备、数据类型及维度检查；`ParamOperator` 管理带量化配置的参数列表。`OpContext` 描述单次执行的设备、CUDA stream、allocator 和 workspace。

`VecAddOp::forward` 先检查三个 Tensor 的设备、数据类型与 shape，再根据 `OpContext::_device_type` 选择 CPU 或 CUDA kernel。当前 Add 只实现 FP32：CPU 路径通过 Armadillo `fvec` 相加，CUDA 路径由每个线程处理一个元素并进行边界检查，也可使用调用方传入的 stream。

现有 `op_test.add` 创建 GPU Tensor、初始化两个输入、直接调用 GPU Add kernel，并将输出复制回主机检查结果；它尚未通过 `VecAddOp::forward`，也没有 CPU 路径测试。

## 5. 关键运行路径

Add 的当前调用路径是：

1. 调用方准备设备、dtype 和 shape 一致的输入与输出 Tensor，并填写 `OpContext`。
2. `VecAddOp::forward` 执行公共 Tensor 检查和 shape 检查。
3. `get_add_kernel` 按 CPU/GPU 返回对应函数指针。
4. CPU kernel 使用 Armadillo 写入输出；CUDA kernel 按元素写入输出，并使用可选 stream 启动。

外部内存路径则由 `Buffer(ptr, capacity, device_type)` 包装；`owns_memory()` 为 false，调用方仍负责外部指针的生命周期。

Operator 的可恢复参数错误通过 `base::Status` 返回；kernel 内部约束和未知设备仍使用 glog `CHECK/LOG(FATAL)`。CUDA 运行失败仍应检查对应 API 或 kernel launch 的错误码。

## 6. 当前边界与已知技术债

以下是阅读和继续开发时最重要的事实，不等同于本次要修复的任务清单：

1. **部分 CUDA 返回值未检查**：部分 memcpy/memset 路径忽略 CUDA API 返回的 `cudaError_t`，失败时可能缺少及时、准确的错误信息；按当前约定可继续使用 `CHECK/LOG(FATAL)` 报错，无需引入额外错误类型。
2. **Add 测试仍是初步覆盖**：现有测试直接调用 CUDA kernel，尚未覆盖 `VecAddOp::forward`、CPU kernel、shape/device/dtype 错误分支、非空 stream 以及无 GPU 环境下的跳过逻辑。
3. **后续模块仍处于规划阶段**：model、runtime、KV cache、模型加载与推理工具尚未接入源码和构建。

## 7. 修改入口速查

| 想做的事情 | 首要入口 | 通常还需同步 |
| --- | --- | --- |
| 增加数据类型/设备类型 | `include/Fire/base/base.h` | dtype 字节数、拷贝/分配逻辑、测试 |
| 修改 CPU/GPU 分配策略 | `include/Fire/base/alloc.h`、`src/base/alloc_*.cpp` | Buffer 行为与 allocator 测试 |
| 修改存储所有权 | `include/Fire/base/buffer.h`、`src/base/buffer.cpp` | Buffer/Tensor 测试 |
| 完善 Tensor | `include/Fire/tensor/tensor.h`、`src/tensor/tensor.cpp` | `src/CMakeLists.txt`、`test/test_tensor/` |
| 增加 Operator | `include/Fire/op/`、`src/op/` | `src/op/kernels/`、`src/CMakeLists.txt`、对应测试目录 |
| 增加模型或 Runtime | 新建对应 public header 与 `src` 子目录 | CMake 源文件、测试、README/本文 |
| 增加测试 | `test/test_<module>/` | `test/CMakeLists.txt` |
| 增加命令行/benchmark 工具 | `tools/` | 顶层 `add_subdirectory(tools)` 与 tools CMake |

新增实现文件时务必显式确认它已进入某个 CMake target；“文件存在”并不意味着会被编译。

## 8. 近期开发计划

当前基础抽象已形成初步闭环，近期按以下顺序推进：

1. 补齐 Base 与 Tensor 测试，包括 allocator、memcpy/memset、Buffer 生命周期、reshape、共享存储和设备迁移等关键路径。
2. 完善 Add 测试，覆盖 `VecAddOp::forward`、CPU/GPU kernel、参数错误分支、CUDA stream 和运行错误检查。
3. 在测试固定现有抽象和调用约束后，复用 `Operator + OpContext + kernel 分派` 结构扩展其他算子，并为每个新算子同步加入 CPU/GPU 测试。
4. 算子集合具备基本覆盖后，再推进 model、runtime、KV cache、模型加载与推理工具。

## 9. 文档维护约定

当出现以下变化时应同步更新本文件：新增顶层模块、CMake target 或外部依赖；模块从“占位”变为“接入构建”；关键数据流或所有权规则改变；已知边界被修复或替换。
