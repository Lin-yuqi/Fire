# Fire 仓库地图

> 本文描述仓库当前实际状态，用于快速定位代码、理解依赖关系和继续开发。最后核对日期：2026-09-04。

## 1. 项目定位与当前阶段

Fire 是一个以学习和实验为目标、从零构建的轻量级 CUDA 推理框架，计划沿着“内存与 Buffer → Tensor → Operator → Model → Runtime → LLM 推理”的方向演进。

当前仓库仍处于基础设施阶段：

- `base` 模块已经接入构建，提供 CPU/GPU 分配器、内存拷贝以及 Buffer 生命周期管理。
- `tensor` 已接入 `Fire::fire`，具备接口和初步实现，并开始覆盖 Buffer 字节偏移与 clone 行为。
- `op` 只有 Layer 元数据骨架，已接入构建但尚无计算逻辑和测试。
- `model`、runtime、kernel、工具程序等仍是规划或空目录，尚无实现。

因此，当前 Base、Tensor 和 Operator 均可链接，但只有 Base 与 Tensor 的少量路径经过测试。

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
│       └── layer.h                # LayerType 与 BaseLayer 骨架
├── src/
│   ├── CMakeLists.txt             # fire 静态库定义；收集 base/op/tensor 源文件
│   ├── base/
│   │   ├── alloc.cpp              # 通用 memcpy/memset 与工厂静态实例
│   │   ├── alloc_cpu.cpp          # malloc/free 后端
│   │   ├── alloc_cu.cpp           # cudaMalloc/cudaFree 后端
│   │   └── buffer.cpp             # Buffer 构造、析构与属性访问
│   ├── tensor/
│   │   └── tensor.cpp             # Tensor 初步实现
│   ├── op/
│   │   └── layer.cpp              # BaseLayer 属性实现
│   └── model/                     # 预留空目录
├── test/
│   ├── CMakeLists.txt             # 单一 fire_tests 测试可执行文件
│   ├── test_base/test_buffer.cpp  # Buffer 自有/外部内存测试
│   ├── test_tensor/test_tensor.cpp# Tensor 字节偏移与 clone 测试
│   └── test_model/                # 预留空目录
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
| Armadillo | 公开传递给 `fire`；当前源码尚未使用 |
| GoogleTest | 仅在 `BUILD_TESTING=ON` 时查找，用于测试 |

构建目标：

```text
Fire (project)
├── fire / Fire::fire             # 静态库；包含 src/base、src/tensor、src/op
└── fire_tests                    # 链接 Fire::fire + GTest::gtest_main
    ├── test_base/test_buffer.cpp
    └── test_tensor/test_tensor.cpp
```

常用命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

当前环境中已有 build 配置；本次核对时构建成功，CTest 发现并通过 6 个测试：2 个 Buffer、1 个 allocator 和 3 个 Tensor 测试。

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

op::BaseLayer ── 持有 LayerType、DataType、DeviceType 和名称
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

### `op`: Layer 元数据骨架

命名空间为 `op`。`LayerType` 已列出 Linear、Embedding、RMSNorm、Matmul、MHA、Softmax、Add、SwiGLU 等预期算子类型；`BaseLayer` 目前只保存名称、Layer 类型、数据类型和设备类型，没有 forward 接口或计算逻辑。

该模块已接入构建，但没有测试。

## 5. 关键运行路径

目前实际可运行的典型路径是：

1. `CPUAllocatorFactory::get_instance()` 获取 CPU allocator。
2. `Buffer(capacity, allocator)` 请求内存并记录所有权与设备类型。
3. 使用 `Buffer::ptr()` 访问内存。
4. Buffer 析构时，若拥有内存则调用创建它的 allocator 释放。

外部内存路径则由 `Buffer(ptr, capacity, device_type)` 包装；`owns_memory()` 为 false，调用方仍负责外部指针的生命周期。

当前项目规模较小，错误处理采用 glog `CHECK/LOG(FATAL)` 快速终止的策略，暂不引入统一的异常、`Status` 或 `Result` 机制。内部约束和运行失败应提供足够明确的错误信息；这一约定不免除对 CUDA API 返回值的检查。

## 6. 当前边界与已知技术债

以下是阅读和继续开发时最重要的事实，不等同于本次要修复的任务清单：

1. **部分 CUDA 返回值未检查**：部分 memcpy/memset 路径忽略 CUDA API 返回的 `cudaError_t`，失败时可能缺少及时、准确的错误信息；按当前约定可继续使用 `CHECK/LOG(FATAL)` 报错，无需引入额外错误类型。
2. **测试覆盖仍较窄**：目前覆盖 CPU Buffer 分配、外部内存所有权、GPU allocator 构造与设备类型、Tensor `from_blob`、字节偏移和 CPU clone；GPU 内存操作、清零、迁移、更多 Tensor 行为与 Layer 均未覆盖。
3. **文档结构略超前**：README 中展示的部分目录、`tools/CMakeLists.txt` 和模块尚不存在或仍为空，应以实际源码和 CMake 为准。

## 7. 修改入口速查

| 想做的事情 | 首要入口 | 通常还需同步 |
| --- | --- | --- |
| 增加数据类型/设备类型 | `include/Fire/base/base.h` | dtype 字节数、拷贝/分配逻辑、测试 |
| 修改 CPU/GPU 分配策略 | `include/Fire/base/alloc.h`、`src/base/alloc_*.cpp` | Buffer 行为与 allocator 测试 |
| 修改存储所有权 | `include/Fire/base/buffer.h`、`src/base/buffer.cpp` | Buffer/Tensor 测试 |
| 完善 Tensor | `include/Fire/tensor/tensor.h`、`src/tensor/tensor.cpp` | `src/CMakeLists.txt`、`test/test_tensor/` |
| 增加 Operator | `include/Fire/op/`、`src/op/` | `src/CMakeLists.txt`、对应测试目录 |
| 增加模型或 Runtime | 新建对应 public header 与 `src` 子目录 | CMake 源文件、测试、README/本文 |
| 增加测试 | `test/test_<module>/` | `test/CMakeLists.txt` |
| 增加命令行/benchmark 工具 | `tools/` | 顶层 `add_subdirectory(tools)` 与 tools CMake |

新增实现文件时务必显式确认它已进入某个 CMake target；“文件存在”并不意味着会被编译。

## 8. 建议的近期开发顺序

结合现有代码，最短闭环是：

1. 补齐 allocator、memcpy/memset 与 Buffer 生命周期测试，并为 CUDA 调用补充返回值检查。
2. 继续建立 Tensor 的 CPU 构造、reshape、共享存储和迁移测试，再补 GPU 测试。
3. 明确 Operator 抽象（输入输出、forward、参数和设备约束），之后再逐个加入 CUDA kernel。
4. 在基础抽象稳定后，再扩展 model、runtime、KV cache、模型加载与推理工具。

## 9. 文档维护约定

当出现以下变化时应同步更新本文件：新增顶层模块、CMake target 或外部依赖；模块从“占位”变为“接入构建”；关键数据流或所有权规则改变；已知边界被修复或替换。
