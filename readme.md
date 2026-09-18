# 🔥 Fire

> A lightweight CUDA inference framework built from scratch for learning, experimentation, and high-performance LLM inference.

**Fire** 是一个从零实现的轻量级 CUDA 推理框架，用于学习和验证大语言模型推理、CUDA Kernel、显存管理与推理系统架构。

> 🚀 **Fire v0.1 即将上线。** 当前版本已经完成 TinyLlama-1.1B 的 FP32 单 token CPU/CUDA forward 主链路，正在补齐聊天入口、参考结果对齐与发布前收尾。

Fire 仍是个人学习项目。v0.1 的目标是交付一条结构清晰、可测试、可在 CPU 与 CUDA 上运行的 TinyLlama 推理基线，而不是提供可替代成熟推理框架的生产级服务。

---

## 🚀 v0.1 发布候选

### 已完成

- `.fire` v1 Writer、TinyLlama exporter、mmap Reader，以及 201 项 canonical tensor 的完整 Loader 校验。
- FP32 Tensor、内存管理、执行上下文与 CPU/CUDA kernel 分派。
- Add、RMSNorm、Matmul、Linear、Embedding、RoPE、SwiGLU、Softmax kernel 与 MHA。
- TinyLlama 22 层完整 forward：Embedding → Attention → 残差 → FFN → 残差 → final RMSNorm → LM Head。
- 连续 K/V Cache 的分配、逐层写入、有效长度提交与会话 reset。
- 真实约 4.1 GiB `.fire` 模型的双 token CPU/GPU forward 集成测试；GPU 测试覆盖非默认 CUDA stream，并检查 CPU/GPU logits 最大绝对误差小于 `1e-2`。

### 发布前收尾

- 接通 tokenizer、sampling 与 `llama_chat` 命令行生成循环。
- 增加与 Hugging Face 参考实现的 logits/token 对齐记录。
- 整理 release 配置、运行示例与已知限制。

### 当前边界

- 仅支持固定 profile：`TinyLlama/TinyLlama-1.1B-Chat-v1.0`。
- 模型执行目前为单序列、单 token、FP32；尚无 batching、量化或 PagedAttention。
- 独立 `SoftmaxOp` 尚未公开；MHA 已在 CPU/CUDA 实现中内置稳定 softmax。
- `demo/llama_chat.cpp` 正在接入，暂不属于已验证的 v0.1 core 路径。

---

## ✨ 项目目标

- 理解 LLM 推理框架从 Tensor 到模型执行的完整数据流。
- 使用 CUDA 实现并验证 Transformer 核心算子。
- 从零实现内存管理、Tensor、Operator、Model 与 Runtime。
- 通过 GoogleTest 和真实模型集成测试建立可回归的正确性基线。
- 在正确性稳定后继续进行量化、Kernel Fusion 与性能分析。

---

## 🏗️ 项目结构

```text
Fire/
├── include/Fire/
│   ├── base/                 # 设备、状态、Allocator 与 Buffer
│   ├── tensor/               # Tensor 元数据、存储与设备迁移
│   ├── op/                   # Operator 公共接口
│   └── model/                # Reader、Loader、KVCache 与 TinyLlama
├── src/
│   ├── base/
│   ├── tensor/
│   ├── op/
│   │   └── kernels/
│   │       ├── cpu/
│   │       └── cuda/
│   └── model/
├── test/
│   ├── test_base/
│   ├── test_tensor/
│   ├── test_op/
│   └── test_model/
├── tools/
│   ├── fire_writer.py
│   └── export_tinyllama.py
├── demo/
│   └── llama_chat.cpp        # v0.1 聊天 CLI 接入中
├── docs/
│   ├── model_export_v0_1.md
│   ├── model_design_v0_1.md
│   └── repo_map.md
├── CONTEXT.md
├── CMakeLists.txt
└── readme.md
```

核心 C++ 代码编译为静态库 `libfire.a`，并通过 CMake target `Fire::fire` 提供给测试和上层程序。

---

## 🧱 已实现模块

### Runtime 基础设施

- `CPUAllocator` / `GPUAllocator` 管理主机和设备内存。
- `Buffer` 表达底层存储、设备位置与所有权。
- `Tensor` 表达 dtype、shape、stride、offset，并支持 clone 和 CPU/CUDA 迁移。
- `OpContext` 逐次传入设备、allocator、CUDA stream 与 workspace，不把执行环境固化在算子或模型中。

### Operator 与 Kernel

当前已有 FP32 CPU/CUDA 路径：

- VecAdd
- RMSNorm
- Matmul / Linear
- Embedding
- RoPE
- SwiGLU
- Softmax kernel
- Multi-Head Attention（含 GQA）

RMSNorm、RoPE、Embedding、SwiGLU 和 MHA 支持调用方提供的 CUDA stream。RoPE 使用与 Hugging Face TinyLlama 权重布局一致的前后半区配对和紧凑 sin/cos cache。

### TinyLlama Model

`TinyLlamaModel` 已实现以下生命周期：

```text
TinyllamaLoader::load_weights
        ↓
TinyLlamaModel::create
        ↓
TinyLlamaModel::prepare(capacity)
        ↓
forward(token_id, pos, logits)
        ↓
reset()
```

一次 `forward` 的主路径为：

```text
token_id
   ↓
Embedding
   ↓
22 × [RMSNorm → Q/K/V → RoPE → KVCache → MHA → Wo → Residual
      → RMSNorm → W1/W3 → SwiGLU → W2 → Residual]
   ↓
Final RMSNorm → LM Head → logits[32000]
```

Runtime 中间 Tensor 会跨层复用；K/V Cache 的形状为
`[num_layers, capacity, num_kv_heads, head_dim]`。只有整次 forward 成功后才提交新的有效序列长度。

更完整的 shape、生命周期和错误语义见
[`docs/model_design_v0_1.md`](docs/model_design_v0_1.md)。

### `.fire` 模型格式

- `FireWriter` 写入 model-agnostic `.fire` v1 容器。
- TinyLlama exporter 对本地 BF16 safetensors 进行两遍校验和逐 Tensor FP32 转换，避免加载完整 `state_dict`。
- `FireReader` 使用 mmap 解析并校验 Header、Directory、payload 范围与布局。
- `TinyllamaLoader` 校验固定 profile 的 201 个 tensor，并建立共享 mmap Tensor views。

格式和 canonical tensor mapping 见
[`docs/model_export_v0_1.md`](docs/model_export_v0_1.md)。

---

## 🔨 构建

主要依赖：

- 支持 C++17 的编译器
- CMake
- CUDA Toolkit
- glog
- Armadillo
- GoogleTest（测试构建）
- Python 3、NumPy、safetensors（导出与合同测试）
- PyTorch（导出真实 TinyLlama payload）

当前 `llama_chat` demo 正在接入。构建已验证的 core 与测试目标：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target fire fire_tests -j
```

---

## 🧪 测试

运行常规测试：

```bash
ctest --test-dir build --output-on-failure
```

或直接运行 C++ 测试：

```bash
./build/test/fire_tests
```

Writer/exporter 合同测试：

```bash
python3 -B test/test_model/test_export_tinyllama.py
```

### 真实 TinyLlama forward

先将真实模型导出到 `tmp/llama.fire`。这两项重型测试默认跳过，需要显式开启：

```bash
# CPU：连续执行两个 token，并验证 logits 与 KV Cache 位置约束
FIRE_RUN_TINYLLAMA_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='TinyllamaTest.CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache'

# GPU：需要 CUDA 和至少 5 GiB 空闲显存；同时比较 CPU/GPU logits
FIRE_RUN_TINYLLAMA_GPU_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='TinyllamaTest.GpuForwardTwoTokensOnNonDefaultStream'
```

2026-09-18 的发布候选验证中，常规 CTest 共 64 项、0 failure，其中 13 项因 CUDA/真实模型开关未启用而跳过；显式开启的真实 GPU 测试已在 RTX 4070 Laptop GPU 上通过，两个位置的 CPU/GPU logits 最大绝对误差分别约为 `1.90e-4` 和 `1.92e-4`，低于测试阈值 `1e-2`。

---

## 📦 导出 TinyLlama

输入目录需包含与 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` profile 精确匹配的 `config.json` 和单个 `model.safetensors`：

```bash
mkdir -p tmp
python3 tools/export_tinyllama.py \
  --hf /path/to/TinyLlama-1.1B-Chat-v1.0 \
  tmp/llama.fire
```

输出路径必须尚不存在。导出结果约为 4.1 GiB；当前 `.fire` v1 只保存 FP32 tensor，不保存 tokenizer 或生成配置。

---

## 🛣️ Roadmap

### v0.1：TinyLlama FP32 推理基线（发布收尾中）

- [x] Memory / Buffer / Tensor
- [x] CPU/CUDA Operator 与 Kernel
- [x] `.fire` v1 Writer、Reader、Exporter 与 Loader
- [x] MHA、KV Cache 与 TinyLlama 完整 forward
- [x] 真实模型 CPU/GPU 双 token 验证
- [ ] Tokenizer、Sampling 与聊天 CLI
- [ ] Hugging Face 参考 logits/token 对齐
- [ ] Release 配置、示例与已知限制整理

### v0.2+

- Batch 与更灵活的模型 profile
- Matmul 量化与权重量化格式
- Kernel Fusion、显存复用与性能优化
- Nsight Systems / Nsight Compute profiling
- PagedAttention 与更完整的生成 runtime

---

## 🚧 Development Status

Fire v0.1 当前属于 **release candidate / 发布收尾阶段**。核心 TinyLlama FP32 forward 已经在 CPU 和 CUDA 上跑通，模型加载、Runtime shape、算子顺序、RoPE、GQA Attention、KV Cache 推进以及最终 logits 均进入自动化测试。

当前仍不能把 Fire 描述为完整聊天引擎：仓库尚未内置 tokenizer 与 sampler，`llama_chat` demo 也仍在连接这些上层组件；同时还需要补充 Hugging Face 参考对齐和更系统的 CUDA 性能数据。完成这些发布项后将发布 **v0.1**。

---

## 🔥 Why Fire?

Fire 希望从一个 CUDA Kernel 开始，逐步构建一条可理解、可验证的推理链路：

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
