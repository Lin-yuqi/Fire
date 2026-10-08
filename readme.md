# 🔥 Fire

> 一个从零实现、面向学习和实验的轻量级 CUDA LLM 推理框架。

**Fire v0.2 已接通 Qwen3 聊天。** 保留 v0.1 的 TinyLlama FP32 基线，新增
Qwen3-0.6B/8B profile、ByteLevel BPE tokenizer 和 `qwen_chat`。本地
Qwen3-8B-AWQ 已完成 GPU 多轮中文生成与 `/reset` 验证。

```text
TinyLlama / Qwen3 checkpoint
        ↓
Model-specific exporter → FP32 .fire v1 / INT4、BF16 .fire v2
        ↓
FireReader / TinyllamaLoader / Qwen3Loader
        ↓
TinyLlamaModel / Qwen3Model forward + KV Cache
        ↓
SentencePiece / Qwen3 ByteLevel BPE tokenizer + Argmax sampler
        ↓
llama_chat / qwen_chat CLI
```

Fire 是个人学习项目。v0.2 在 TinyLlama 基线上补齐 Qwen3 文本生成入口，保持
单序列、逐 token 的执行方式，便于阅读和测试。

## v0.2 能做什么

- 读取固定 TinyLlama profile 的 BF16 Hugging Face safetensors，并导出为 FP32
  `.fire` v1 Tensor Container。
- 通过 mmap 读取 `.fire`，校验并绑定 201 个 canonical tensor。
- 在 CPU 或 CUDA 上执行完整 TinyLlama forward：Embedding、22 层
  Attention/FFN、RoPE、GQA、KV Cache、final RMSNorm 和 LM Head。
- 使用 SentencePiece 完成文本编解码，使用 ArgmaxSampler 做 greedy decoding。
- 运行一个简单的多轮聊天 CLI；跨轮复用 KV Cache，达到 2048 token 上限后结束会话。
- 通过同一套 Qwen3Loader/Qwen3Model 执行 0.6B/8B profile；FP32 权重支持
  CPU/CUDA，INT4 权重执行要求 GPU。
- 将官方 Qwen3-8B-AWQ GEMM 权重导入 `.fire` v2；Embedding/LM Head 保留
  BF16，投影使用 Fire 的 INT4 布局，激活和 KV Cache 使用 FP32。
- 使用 Qwen3 tokenizer 和聊天模板进行多轮生成，支持 thinking 开关、流式 UTF-8
  输出、上下文预算、两种 EOS 和 `/reset`。
- 通过 GoogleTest 和 Python contract tests 验证 Buffer、Tensor、Operator、Reader、
  exporter、tokenizer、sampler 和模型运行路径。

当前明确不支持 batching、随机采样、PagedAttention、通用模型自动识别或
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
- nlohmann_json CMake package / headers
- ICU development headers/libraries（uc、i18n）
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
  nlohmann-json3-dev \
  libicu-dev \
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
build/demo/llama_chat     # TinyLlama 聊天 CLI
build/demo/qwen_chat      # Qwen3 聊天 CLI
```

如果只想构建库和 demo、不安装 GoogleTest：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF
cmake --build build --target fire llama_chat qwen_chat --parallel
```

重新开启测试时需要重新配置：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
```

CMake 会在构建目录中为选中的 ICU 建立独立头文件入口，并优先用于 Fire 编译，
避免 Conda 的 JSON 包引入其他版本的 ICU 头文件。正常配置无需手动指定
`nlohmann_json_DIR`；已有构建目录直接重新运行 `cmake -S . -B build` 即可。

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

#### Qwen3 exporter（v0.2）

同一个 exporter 会从 `config.json` 自动识别 Qwen3-0.6B 或 Qwen3-8B，并支持单个
`model.safetensors` 或标准 HF shard index：

```bash
python -B tools/export_qwen3.py \
  --hf models/Qwen3-0.6B \
  tmp/qwen3-0.6b.fire
```

0.6B 会校验 311 个 BF16 source tensor，生成预期长度为 `3,006,559,424` bytes
（约 2.80 GiB）的 FP32 `.fire` v1。当前本地 0.6B 文件已通过 Loader、两 token
CPU/GPU forward 和 Hugging Face eager attention 的逐位置 logits 对齐；GPU 测试使用非默认
CUDA stream。8B 的相同 FP32 路径会生成约 30.5 GiB 的文件。

本地已量化的 Qwen3-8B-AWQ 使用独立的导入模式，转换 packing 而不再量化投影：

```bash
python -B tools/export_qwen3.py \
  --hf models/Qwen3-8B-AWQ \
  --quantization awq \
  tmp/qwen3-8B-awq.fire
```

该路径生成混合 BF16/INT4 的 `.fire` v2；运行时使用 Qwen3Model 的 GPU 执行路径。
它已通过原始 AWQ/Hugging Face FP32 两 token logits 对齐和 2048 token
teacher-forcing 验证；这些检查与聊天 smoke test 不代替完整的语料质量评估。

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

#### Qwen3 聊天

默认加载 `tmp/qwen3-8B-awq.fire` 和 `models/Qwen3-8B-AWQ/` 的 tokenizer：

```bash
./build/demo/qwen_chat
./build/demo/qwen_chat --help
```

默认使用 8B profile、GPU、2048 token 上下文、128 token 回复上限，关闭 thinking。
支持与 TinyLlama demo 相同的 `/help`、`/reset`、`/exit` 和 `/quit` 命令。

```bash
# 使用自己导出的 0.6B FP32 文件；FP32 也可以选择 cpu
./build/demo/qwen_chat --profile 0.6b \
  tmp/qwen3-0.6b.fire models/Qwen3-0.6B gpu

# 显式启用 thinking，并调整生成预算
./build/demo/qwen_chat --thinking --max-new-tokens 256 --context-size 2048
```

`qwen_chat` 不添加 BOS，按本地 Qwen3 无工具模板构造消息。它只在 tokenizer 有效
词表范围内做 greedy 采样，遇到 `<|im_end|>` 或 `<|endoftext|>` 停止，随后补齐
assistant 的 `<|im_end|>\n`。预先为结束标记与换行留出空间；历史和输入过长时
保留当前对话，并提示缩短输入或 `/reset`。

下一轮模板的 token 前缀与现有缓存一致时复用 KV Cache；移除旧 thinking 内容等
模板变化会触发 reset 并重放历史。输入仍是单行文本，不支持工具调用或滑动窗口。
INT4/AWQ 模型必须选择 GPU；启动时会检查可用显存。

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
ctest --test-dir build -R 'Tokenizer|BPETest|QwenChatTest|fire_qwen_chat_|ArgmaxSampler' --output-on-failure
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

Qwen3-0.6B 的 CPU/GPU forward smoke test 和 Hugging Face logits 对齐分别为：

```bash
FIRE_RUN_QWEN3_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='Qwen3ModelTest.CpuForwardTwoTokensProducesFiniteLogitsAndAdvancesCache'

FIRE_RUN_QWEN3_GPU_FORWARD_TEST=1 \
  ./build/test/fire_tests \
  --gtest_filter='Qwen3ModelTest.GpuForwardTwoTokensMatchesCpuOnNonDefaultStream'

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
│   ├── tokenizer/            # Tokenizer、SentencePiece 与 Qwen3 ByteLevel BPE
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
│   ├── llama_chat.cpp
│   ├── qwen_chat.cpp
│   └── qwen_chat_utils.h
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

- 模型范围是固定 TinyLlama profile 与 Qwen3-0.6B/8B dense profile。
- 模型执行是单序列、单 token；激活和 KV Cache 保持 FP32，没有 batching。
- 两个聊天 demo 只有 greedy sampling，不支持滑动窗口或历史压缩。
- INT4/AWQ 执行只支持 GPU。Qwen3 默认容量为 2048；更大的容量需要单独测量显存。
- CPU 推理仅适合作为正确性基线，速度很慢。
- 项目配置阶段即要求 CUDA Toolkit；尚未提供纯 CPU-only build。
- 部分底层约束仍通过 glog `CHECK/LOG(FATAL)` 处理，不适合作为不可信输入服务边界。
- 已有固定 token 的 Hugging Face logits 对齐；更完整的语料质量回归与量化质量评估
  仍需补充，不能把短提示词生成视为完整模型质量验收。
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

### v0.2：Qwen3 文本生成

- [x] 0.6B/8B 共用的 profile、Loader 和 Qwen3Model
- [x] `.fire` v2 INT4/BF16 与官方 8B AWQ 导入
- [x] Qwen3 ByteLevel BPE tokenizer 与 HF 编码样例对齐
- [x] `qwen_chat`、thinking 开关、上下文管理与 UTF-8 流式输出
- [x] 本地 8B AWQ 两 token HF logits 对齐、2048 token teacher forcing
- [x] 本地 8B AWQ 多轮中文生成与 `/reset`

### 下一阶段：算子优化与吞吐提升

以当前单序列 Qwen3-8B-AWQ GPU 路径为基线，先测量瓶颈，再逐项优化 CUDA
算子。主要目标是提高 decode 的 tokens/s、降低 TPOT，同时观察 TTFT 和显存
占用；以下工作尚未实现，优化效果以实测为准。

- [ ] 建立可重复的性能基线：固定模型、提示词、上下文长度和生成长度，预热后
  多次测量，记录 TTFT、TPOT、decode tokens/s 和峰值显存；分别观察 prompt
  prefill 与逐 token decode，并记录多轮聊天中历史重放的开销。
- [ ] 使用 CUDA events、Nsight Systems / Nsight Compute 定位耗时，区分算子
  执行、kernel launch、CPU/GPU 数据传输和同步开销，再确定优化顺序。
- [ ] 优先尝试 Linear 的单 token 路径：为 `N=1` 设计 INT4 GEMV，改进权重读取、
  解包、group metadata 复用和并行归约；评估 FP32/BF16 GEMV 对 LM Head 等
  dense projection 的收益。
- [ ] 根据 profiling 结果优化 MHA 的 KV Cache 访问与归约，以及 RMSNorm、
  SwiGLU 等算子；尝试融合能减少中间读写和 launch 次数的操作。
- [ ] 每项优化都对照 CPU/原 kernel 与 HF logits 验证数值，覆盖模型实际 shape
  和非默认 CUDA stream，再比较算子耗时、端到端吞吐及显存；保留性能结果与
  已知适用范围。

### 后续验证与优化

- Hugging Face 参考 logits/token 独立对齐
- 更完整的 Loader/状态失败矩阵与 CUDA 错误传播
- temperature、top-k、top-p 等 sampling 策略
- Batch、量化质量回归与更灵活的 Model Profile
- 更进一步的显存复用与生成执行优化
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
