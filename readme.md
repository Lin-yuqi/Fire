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
│   └── model/
├── src/
│   ├── base/
│   ├── model/
│   └── CMakeLists.txt
├── test/
│   ├── test_base/
│   ├── test_model/
│   └── CMakeLists.txt
├── tools/
│   └── CMakeLists.txt
├── CMakeLists.txt
└── README.md
```

核心代码编译为：

```text
Fire::fire
```

对应静态库：

```text
libfire.a
```

测试和工具程序通过链接 `Fire::fire` 使用框架功能。

---

## 📁 模块说明

### `include/Fire`

存放对外提供的头文件：

```cpp
#include "Fire/base/xxx.h"
#include "Fire/model/xxx.h"
```

### `src`

存放核心实现代码，后续将逐步加入：

```text
base/
op/
model/
runtime/
tokenizer/
```

其中 CUDA Kernel 也属于 `Fire::fire`。

### `test`

使用 GoogleTest 对各模块进行测试，最终生成：

```text
fire_tests
```

### `tools`

存放推理、Benchmark、模型转换和 Profiling 等独立工具。

---

## 🧩 构建系统

Fire 使用 CMake 管理项目，主要依赖：

* CUDA
* GoogleTest
* glog
* Armadillo

构建结构：

```text
CMakeLists.txt
├── src/CMakeLists.txt
├── test/CMakeLists.txt
└── tools/CMakeLists.txt
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

计划实现的功能包括：

* RMSNorm、RoPE、Softmax
* GEMV / GEMM、SwiGLU
* Attention、KV Cache
* Model Loader、Tokenizer
* Sampling
* Kernel Fusion、CUDA Stream
* Nsight Compute Profiling

---

## 🚧 Development Status

> **Fire is under active development.**

项目目前处于基础架构开发阶段，模块和 Roadmap 将持续更新。

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
