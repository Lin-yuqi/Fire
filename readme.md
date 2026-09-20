# 🔥 Fire

> 一个从零实现、面向学习和实验的轻量级 CUDA LLM 推理框架。

**Fire v0.1 已发布。** 当前版本围绕固定的
[`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
Model Profile，打通了以下端到端路径：

```text
Hugging Face checkpoint
        ↓
TinyLlama exporter → FP32 .fire v1
        ↓
FireReader / TinyllamaLoader
        ↓
TinyLlamaModel CPU/CUDA forward + KV Cache
        ↓
SentencePiece tokenizer + Argmax sampler
        ↓
llama_chat CLI
```

Fire 是个人学习项目。v0.1 的目标是提供一条结构清晰、可以阅读和测试的
TinyLlama FP32 推理基线，不是生产级推理服务，也不试图替代 llama.cpp、vLLM
或 Transformers。

## v0.1 能做什么

- 读取固定 TinyLlama profile 的 BF16 Hugging Face safetensors，并导出为 FP32
  `.fire` v1 Tensor Container。
- 通过 mmap 读取 `.fire`，校验并绑定 201 个 canonical tensor。
- 在 CPU 或 CUDA 上执行完整 TinyLlama forward：Embedding、22 层
  Attention/FFN、RoPE、GQA、KV Cache、final RMSNorm 和 LM Head。
- 使用 SentencePiece 完成文本编解码，使用 ArgmaxSampler 做 greedy decoding。
- 运行一个简单的多轮聊天 CLI；跨轮复用 KV Cache，达到 2048 token 上限后结束会话。
- 通过 GoogleTest 和 Python contract tests 验证 Buffer、Tensor、Operator、Reader、
  exporter、tokenizer、sampler 和模型运行路径。

当前明确不支持 batching、量化、随机采样、PagedAttention、通用模型自动识别或
生产服务。完整限制见[已知限制](#已知限制)。

## 快速开始

下面的命令假设使用 64-bit little-endian Linux，并在仓库根目录执行。Ubuntu/Debian
可以直接参考；其他发行版请安装对应的开发包。

### 1. 安装 C++ 构建依赖

最低要求：

- CMake 3.17+
- 支持 C++17 的编译器
- NVIDIA CUDA Toolkit 和 `nvcc`
- glog
- Armadillo
- SentencePiece development headers/library
- GoogleTest（只在构建测试时需要）
- Python 3.9+（导出和 Python tests）

Ubuntu/Debian 示例：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  libgoogle-glog-dev \
  libarmadillo-dev \
  libsentencepiece-dev \
  libgtest-dev \
  python3 \
  python3-pip \
  python3-venv
```

CUDA Toolkit 请按 NVIDIA 对应平台的安装方式安装，并确认：

```bash
nvcc --version
```

当前顶层 CMake 以 `LANGUAGES CXX CUDA` 配置项目，因此即使只准备运行 CPU
推理，也必须能找到 CUDA compiler 和 CUDAToolkit。源码还使用了
`/usr/local/cuda/targets/x86_64-linux/include/cccl`；非默认 CUDA 安装需要提供兼容
路径或调整 CMake 配置。

### 2. 配置并构建

推荐使用 Release 构建。包含测试的完整构建：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build --parallel
```

主要产物：

```text
build/src/libfire.a       # Fire 静态库
build/test/fire_tests     # C++ GTest
build/demo/llama_chat     # 聊天 CLI
```

如果只想构建库和 demo、不安装 GoogleTest：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF
cmake --build build --target fire llama_chat --parallel
```

重新开启测试时需要重新配置：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
```

### 3. 准备 Python 导出环境

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy safetensors torch huggingface_hub
```

`torch` 只用于 exporter 在 CPU 上读取 BF16 tensor 并转换为 FP32；聊天 demo
本身不依赖 Python 或 PyTorch。若需要特定 PyTorch wheel，请使用
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。

### 4. 下载 TinyLlama

仓库约定把 Source Checkpoint 放到：

```text
models/TinyLlama-1.1B-Chat-v1.0/
```

使用 Hugging Face 当前的 `hf download` CLI：

```bash
mkdir -p models/TinyLlama-1.1B-Chat-v1.0

hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  config.json \
  model.safetensors \
  tokenizer.model \
  --local-dir models/TinyLlama-1.1B-Chat-v1.0
```

这三个文件分别用于 profile 校验、权重导出和聊天分词。完成后目录至少应为：

```text
models/TinyLlama-1.1B-Chat-v1.0/
├── config.json
├── model.safetensors
└── tokenizer.model
```

`hf download` 和 `--local-dir` 的当前行为见
[Hugging Face 官方下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。
模型为公开仓库，正常情况下无需登录；受限网络环境可能需要配置 Hugging Face
mirror/proxy。

`models/` 已被 `.gitignore` 忽略，不会误提交数 GB 的权重文件。

### 5. 导出 `tmp/llama.fire`

Fire v0.1 不直接读取 Hugging Face safetensors。先将 checkpoint 导出到仓库约定位置：

```bash
mkdir -p tmp

python -B tools/export_tinyllama.py \
  --hf models/TinyLlama-1.1B-Chat-v1.0 \
  tmp/llama.fire
```

exporter 会：

1. 校验 `config.json` 是否精确匹配 v0.1 TinyLlama profile；
2. 校验单个 `model.safetensors` 中 201 个 tensor 的名称、BF16 dtype 和 shape；
3. 将 tensor 逐个转换为 FP32；
4. 写出 `.fire` v1 Header、Directory 和连续 payload。

成功结果的固定属性：

```text
路径:       tmp/llama.fire
tensor 数:  201
data_offset: 19,328 bytes
文件大小:   4,400,212,864 bytes（约 4.10 GiB）
```

可以核对文件大小：

```bash
stat -c '%n %s bytes' tmp/llama.fire
```

输出路径必须尚不存在，exporter 不会覆盖已有文件。需要重新导出时，请先把旧文件移动
到备份位置，或选择一个新的输出文件名。建议至少准备约 8 GiB 可用磁盘空间以同时保存
源 checkpoint 与 FP32 `.fire`。

`.fire` 只保存模型 tensor，不包含 tokenizer；启动 demo 时仍需要原始
`tokenizer.model`。`tmp/` 同样已被 `.gitignore` 忽略。

#### Qwen3 exporter（v0.2 开发中）

同一个 exporter 会从 `config.json` 自动识别 Qwen3-0.6B 或 Qwen3-8B，并支持单个
`model.safetensors` 或标准 HF shard index：

```bash
python -B tools/export_qwen3.py \
  --hf models/Qwen3-0.6B \
  tmp/qwen3-0.6b.fire
```

0.6B 会校验 311 个 BF16 source tensor，生成预期长度为 `3,006,559,424` bytes
（约 2.80 GiB）的 FP32 `.fire` v1。当前本地 0.6B 文件已通过 Loader 和两 token CPU
forward，并与 Hugging Face eager attention 的逐位置 logits 对齐。Qwen3 的 GPU forward、
tokenizer/chat 入口和量化执行仍未验收。8B 的相同路径会生成约 30.5 GiB 的 FP32 文件；
Fire v1 目前没有量化 payload 编码，不应把它当作最终 8B 量化方案。

### 6. 启动聊天 demo

CMake 会把上述默认路径编译进 demo，默认使用 GPU：

```bash
./build/demo/llama_chat
```

也可以显式传入模型路径、tokenizer 路径和设备。CPU 可以运行，但 1.1B FP32 模型会很慢：

```bash
./build/demo/llama_chat \
  tmp/llama.fire \
  models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
  cpu
```

查看参数：

```bash
./build/demo/llama_chat --help
```

CLI 内支持：

```text
/help   查看命令
/reset  清空对话历史
/exit   退出（/quit 也可以）
```

当前 demo 的行为边界：

- 使用 TinyLlama chat template，并显式插入 BOS/EOS token；
- 使用 greedy argmax，不支持 temperature、top-k 或 top-p；
- 每轮最多生成 128 token；
- 模型上下文上限为 2048 token；空间不足时会缩短本轮回复上限并预留 assistant EOS；
- system prompt 只在首轮写入，后续只追加新 user/assistant token，并持续复用同一 KV Cache；
- 不裁剪或重放历史；上下文写满后会提示并结束当前会话，`/reset` 可在写满前手动开始新会话；
- 输入是单行文本。

真实 GPU forward 测试要求至少约 5 GiB 空闲显存；demo 的实际需求还会受到 CUDA
runtime、allocator cache 和设备上其他进程影响。

## 测试

完成构建、模型下载和导出后运行全部已注册测试：

```bash
ctest --test-dir build --output-on-failure
```

直接运行 C++ GTest：

```bash
./build/test/fire_tests
```

只运行 tokenizer 或 sampler：

```bash
ctest --test-dir build -R 'LlamaTokenizer|ArgmaxSampler' --output-on-failure
```

Python Writer/exporter contract tests：

```bash
python3 -B test/test_model/test_export_tinyllama.py
python3 -B test/test_model/test_export_qwen3.py
```

真实模型的双 token forward 属于重型集成测试，默认不会运行：

```bash
# CPU
FIRE_RUN_TINYLLAMA_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='TinyllamaTest.CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache'

# GPU；还会与 CPU logits 比较
FIRE_RUN_TINYLLAMA_GPU_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='TinyllamaTest.GpuForwardTwoTokensOnNonDefaultStream'
```

Qwen3-0.6B 的 CPU forward smoke test 和 Hugging Face logits 对齐分别为：

```bash
FIRE_RUN_QWEN3_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='Qwen3ModelTest.CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache'

FIRE_RUN_QWEN3_HF_ALIGNMENT_TEST=1 \
  ctest --test-dir build -R '^fire_qwen3_hf_logits_alignment$' --output-on-failure
```

没有 CUDA device/driver 时，CUDA tests 会跳过。部分真实模型测试还会根据环境变量、
`tmp/llama.fire` 是否存在和可用显存决定是否跳过。

## 项目结构

```text
Fire/
├── include/Fire/
│   ├── base/                 # Status、设备、Allocator、Buffer
│   ├── tensor/               # Tensor shape、view 与存储
│   ├── op/                   # Operator 公共接口
│   ├── model/                # Reader、Loader、KVCache、TinyLlama、Qwen3 框架
│   ├── tokenizer/            # Tokenizer 接口与 Llama SentencePiece 实现
│   └── sampler/              # Sampler 接口与 ArgmaxSampler
├── src/
│   ├── base/
│   ├── tensor/
│   ├── op/kernels/{cpu,cuda}/
│   ├── model/
│   ├── tokenizer/
│   └── sampler/
├── test/                     # C++ GTest 与 Python contract tests
├── tools/
│   ├── fire_writer.py
│   ├── export_tinyllama.py
│   └── export_qwen3.py
├── demo/
│   └── llama_chat.cpp
├── docs/
│   ├── model_export_v0_1.md
│   ├── model_design_v0_1.md
│   └── repo_map.md
├── CONTEXT.md
├── CMakeLists.txt
└── readme.md
```

核心 C++ 代码编译为静态库 `libfire.a`，并通过 CMake target `Fire::fire` 提供给
测试和 demo。

## v0.1 技术概览

一次 `TinyLlamaModel::forward` 消费一个 token，并产生下一 token 的
`logits[32000]`：

```text
token_id
   ↓
Embedding
   ↓
22 × [RMSNorm → Q/K/V → RoPE → KVCache → GQA MHA → Wo → Residual
      → RMSNorm → W1/W3 → SwiGLU → W2 → Residual]
   ↓
Final RMSNorm → LM Head → logits[32000]
```

Runtime Tensor 跨层复用。K/V Cache 使用连续布局
`[num_layers, capacity, num_kv_heads, head_dim]`，只有整个 token forward 成功后才
提交新的有效长度。

`.fire` 是 model-agnostic 的 Tensor Container，不是自描述的通用模型格式。
TinyLlama 的名称、shape 和配置约束由 `TinyllamaLoader` 与固定 Model Profile 负责。

进一步阅读：

- [模型层设计](docs/model_design_v0_1.md)
- [模型导出与 `.fire` v1 格式](docs/model_export_v0_1.md)
- [仓库地图](docs/repo_map.md)
- [领域术语](CONTEXT.md)

## 已知限制

- v0.1 只支持 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 这一固定 profile。
- 模型执行是单序列、单 token、FP32；没有 batching 和量化执行路径。
- `llama_chat` 只有 greedy sampling，KV Cache 只追加、不支持滑动窗口或历史压缩。
- CPU 推理仅适合作为正确性基线，速度很慢。
- 项目配置阶段即要求 CUDA Toolkit；尚未提供纯 CPU-only build。
- 部分底层约束仍通过 glog `CHECK/LOG(FATAL)` 处理，不适合作为不可信输入服务边界。
- 尚未形成与 Hugging Face 参考实现逐位置 logits/token 的独立对齐记录；已有验证主要是
  CPU/GPU 自一致性、算子数值测试和端到端生成路径。
- Softmax 已有 CPU/CUDA kernel 并用于 MHA，但尚无独立公开 `SoftmaxOp`。

## Roadmap

### v0.1：TinyLlama FP32 推理基线

- [x] Memory / Buffer / Tensor
- [x] CPU/CUDA Operator 与 Kernel
- [x] `.fire` v1 Writer、Reader、Exporter 与 Loader
- [x] MHA、KV Cache 与 TinyLlama 完整 forward
- [x] SentencePiece tokenizer 与 Argmax sampler
- [x] 简单多轮 `llama_chat` CLI
- [x] 真实模型 CPU/GPU 双 token 验证

### v0.1 后续验证与 v0.2+

- Hugging Face 参考 logits/token 独立对齐
- 更完整的 Loader/状态失败矩阵与 CUDA 错误传播
- temperature、top-k、top-p 等 sampling 策略
- Batch、量化与更灵活的 Model Profile
- Kernel Fusion、显存复用和 profiling
- PagedAttention 与更完整的生成 runtime

## Why Fire?

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
