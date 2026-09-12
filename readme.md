# 🔥 Fire

> A lightweight CUDA inference framework built from scratch for learning, experimentation, and high-performance LLM inference.

**Fire** 是一个从零实现的轻量级 CUDA 推理框架，旨在帮助理解大语言模型推理、CUDA Kernel、显存管理和推理系统架构。

项目目前处于早期阶段，将按照：

**基础设施 → Tensor → Operator → Model → Runtime → LLM Inference**

逐步完善。

---

## ✨ 项目目标

* 🧠 理解 LLM 推理框架的工作流程
* ⚡ 使用 CUDA 实现高性能算子
* 🧱 从零实现 Tensor、内存管理和 Operator
* 🚀 支持 Transformer / LLaMA 类模型推理
* 🧪 使用 GoogleTest 进行测试
* 🛠️ 使用 CMake 管理项目
* 📊 进行 Kernel Profiling 和性能优化

Fire 更关注理解框架设计，而不仅仅是快速实现功能。

---

## 🏗️ 项目结构

```text
Fire/
├── include/Fire/
│   ├── base/
│   ├── tensor/
│   ├── op/
│   └── model/
├── src/
│   ├── base/
│   ├── tensor/
│   ├── op/
│   │   └── kernels/
│   │       ├── cpu/
│   │       └── cuda/
│   ├── model/
│   └── CMakeLists.txt
├── test/
│   ├── test_base/
│   ├── test_tensor/
│   ├── test_op/
│   ├── test_model/
│   ├── utils.cu
│   └── CMakeLists.txt
├── tools/
│   ├── fire_writer.py
│   └── export_tinyllama.py
├── docs/
│   ├── model_export_v0_1.md
│   └── repo_map.md
├── CONTEXT.md
├── CMakeLists.txt
└── readme.md
```

核心代码编译为：

```text
Fire::fire
```

对应静态库：

```text
libfire.a
```

C++ 测试通过链接 `Fire::fire` 使用框架功能；Python Writer/exporter 作为独立工具运行。

---

## 📁 模块说明

### `include/Fire`

存放对外提供的头文件：

```cpp
#include "Fire/base/xxx.h"
#include "Fire/tensor/xxx.h"
#include "Fire/op/xxx.h"
#include "Fire/model/fire_reader.h"
```

### `src`

存放核心实现代码。当前已包含：

```text
base/
tensor/
model/
op/
└── kernels/
    ├── cpu/
    └── cuda/
```

其中 CPU/CUDA Kernel 和 `.fire` v1 `FireReader` 均编入 `Fire::fire`。
当前已接入 FP32 Add 与 RMSNorm；RMSNorm 支持自定义 epsilon、
CUDA stream，以及按最后一维处理二维输入。

### `test`

使用 GoogleTest 对 C++ 模块进行测试，最终生成：

```text
fire_tests
```

CMake 会在构建 `fire_tests` 前用独立 stdlib Python 脚本生成
260-byte `.fire` v1 fixture；CTest 还会运行 Writer/TinyLlama exporter 的
Python 合同测试。

### `tools`

当前包含 model-agnostic `.fire` v1 `FireWriter` 和 model-specific TinyLlama
两遍 exporter。Exporter 将精确匹配 profile 的本地 BF16 safetensors
逐 tensor 转为 FP32，不加载完整 `state_dict`。

---

## 🧩 构建系统

Fire 使用 CMake 管理项目，主要依赖：

* CUDA
* GoogleTest
* glog
* Armadillo
* Python 3（fixture、合同测试与 exporter CLI）
* NumPy、safetensors（Writer/exporter 合同测试与实际导出）
* PyTorch（真实 TinyLlama payload 导出）

构建结构：

```text
CMakeLists.txt
├── src/CMakeLists.txt
└── test/CMakeLists.txt
```

---

## 🔨 Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

---

## 🧪 Test

```bash
ctest --test-dir build --output-on-failure
```

或直接运行：

```bash
./build/test/fire_tests
```

只运行 RMSNorm 测试：

```bash
./build/test/fire_tests --gtest_filter='rmsnorm_test.*:RmsNormCudaTest.*'
```

只运行 `.fire` Reader 合同测试：

```bash
./build/test/fire_tests --gtest_filter='FireReaderTest.*'
```

单独运行 Writer/exporter 合同测试：

```bash
python3 -B test/test_model/test_export_tinyllama.py
```

---

## 📦 TinyLlama 导出

输入目录需包含精确匹配 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
profile 的 `config.json` 和单个 `model.safetensors`：

```bash
python3 tools/export_tinyllama.py --hf /path/to/TinyLlama /path/to/model.fire
```

输出路径必须不存在。格式、固定 tensor mapping 和验收边界详见
[`docs/model_export_v0_1.md`](docs/model_export_v0_1.md)。

---

## 🛣️ Roadmap

```text
Memory / Buffer
      ↓
Tensor
      ↓
CUDA Runtime
      ↓
Operator
      ↓
CUDA Kernels
      ↓
Transformer Operators
      ↓
KV Cache
      ↓
Model Loader
      ↓
Tokenizer
      ↓
LLaMA-like Model
      ↓
Autoregressive Inference
```

计划实现或继续完善的功能包括：

* RoPE、Softmax、SwiGLU
* TinyLlama 全量导出/回读验收与 ModelLoader
* MatMul/GEMV/GEMM 及相关量化路径
* Attention、KV Cache
* Tokenizer
* Sampling
* Kernel Fusion、CUDA Stream
* Nsight Compute Profiling

---

## 🚧 Development Status

> **Fire is under active development.**

项目已经形成第一版基础抽象：内存管理、Tensor、Operator、执行上下文与
CPU/CUDA kernel 分派均已接入构建。向量 Add 已具备 CPU/CUDA FP32 实现；
RMSNorm 已具备 CPU 一维和 CUDA 一维/二维 FP32 路径，并覆盖自定义 epsilon、
非默认 stream、非 4 整数倍宽度及参数错误测试。RMSNorm 的量化权重尚不支持。

`.fire` v1 的 Writer、TinyLlama-specific exporter、mmap FireReader 与自动化
wire/profile 合同测试已完成。真实 checkpoint 的 metadata 已确认为
201 项，但完整 4.4 GB FP32 导出、FireReader 全量解析与选定元素
bit-exact 回读尚未执行，因此 Export Compatibility 仍未宣称完成。

下一阶段先完成该真实导出/回读里程碑，再实现 TinyLlama ModelLoader
并在真实 shape 上建立 MatMul 基线。完整 Transformer 和 Runtime 仍在开发中。

---

## 🔥 Why Fire?

Fire 希望从一个 CUDA Kernel 开始，逐步构建完整的推理系统：

```text
One Kernel
    ↓
One Operator
    ↓
One Layer
    ↓
One Model
    ↓
One Inference Engine
```

**Build it. Understand it. Optimize it. 🔥**
