# 🔥 Fire

> 一个从零实现、面向学习和实验的轻量级 CUDA LLM 推理框架。

**Fire v0.2 已接通 Qwen3 聊天。** 保留 v0.1 的 TinyLlama FP32 基线，新增
Qwen3-0.6B/8B profile、ByteLevel BPE tokenizer 和 `qwen_chat`。本地
Qwen3-8B-AWQ 已完成 GPU 多轮中文生成与 `/reset` 验证，并接入 INT4、BF16/FP32
单 token CUDA GEMV。固定负载的性能记录见 [基线与优化复测](docs/baseline.md)。

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
- 通过 mmap 读取 `.fire`，由模型专用 Loader 校验名称、dtype 和 shape；TinyLlama
  的 FP32 profile 包含 201 个 canonical tensor。
- 在 CPU 或 CUDA 上执行完整 TinyLlama forward：Embedding、22 层
  Attention/FFN、RoPE、GQA、KV Cache、final RMSNorm 和 LM Head。
- 使用 SentencePiece 完成文本编解码，使用 ArgmaxSampler 做 greedy decoding。
- 运行一个简单的多轮聊天 CLI；跨轮复用 KV Cache，达到 2048 token 上限后结束会话。
- 通过同一套 Qwen3Loader/Qwen3Model 执行 0.6B/8B profile；FP32 权重支持
  CPU/CUDA，INT4 权重执行要求 GPU。
- 将官方 Qwen3-8B-AWQ GEMM 权重导入 `.fire` v2；Embedding/LM Head 保留
  BF16，投影使用 Fire 的 INT4 布局，激活和 KV Cache 使用 FP32。
- 从未量化 Qwen3 checkpoint 执行 RTN INT4 导出；这条路径的 Embedding/Norm
  为 FP32，投影与 LM Head 为 INT4，与 AWQ 的混合布局分开处理。
- 为 CUDA 单 token Linear 提供 INT4、FP32/BF16 GEMV：warp 按行计算、向量化
  加载与归约；不满足专用条件时保留标量或通用矩阵乘路径。
- 使用 Qwen3 tokenizer 和聊天模板进行多轮生成，支持 thinking 开关、流式 UTF-8
  输出、上下文预算、两种 EOS 和 `/reset`。
- 通过 GoogleTest 和 Python contract tests 验证 Buffer、Tensor、Operator、Reader、
  exporter、tokenizer、sampler 和模型运行路径。

当前明确不支持 batching、随机采样、PagedAttention、通用模型自动识别或
生产服务。完整限制见[已知限制](#已知限制)。

## 快速开始

下面的命令假设使用 64-bit little-endian Linux，并在仓库根目录执行。Ubuntu/Debian
可以直接参考；其他发行版请安装对应的开发包。

TinyLlama 和 Qwen3 的权重可以按需准备。如果只运行默认的 `qwen_chat`，下载
Qwen3-8B-AWQ 并执行对应的 AWQ 导入命令即可，无需先下载或导出 TinyLlama。

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

`torch` 用于 exporter 在 CPU 上读取 tensor、转换 FP32 或保留 BF16 位模式；
独立的 Hugging Face 对齐测试也使用它。聊天 demo 本身不依赖 Python 或 PyTorch。
若需要特定 PyTorch wheel，请使用
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。

### 4. 下载模型

#### TinyLlama

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

#### Qwen3

默认聊天与本文性能记录使用官方 8B AWQ checkpoint：

```bash
hf download Qwen/Qwen3-8B-AWQ \
  --include '*.json' --include '*.safetensors' \
  --local-dir models/Qwen3-8B-AWQ
```

若希望先运行较小的 FP32 基线，可以改为下载 0.6B：

```bash
hf download Qwen/Qwen3-0.6B \
  --include '*.json' --include '*.safetensors' \
  --local-dir models/Qwen3-0.6B
```

Qwen3 exporter 需要 `config.json` 和单个 safetensors 文件，或 shard index 及其
引用的全部分片；tokenizer 需要 `tokenizer.json` 与 `tokenizer_config.json`。
上述过滤会下载这些 JSON 与权重文件。参数用法见
[Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)。

`models/` 已被 `.gitignore` 忽略，不会误提交数 GB 的权重文件。

### 5. 导出 Fire 权重

两种模型共用默认输出目录，导出前先创建：

```bash
mkdir -p tmp
```

#### TinyLlama

Fire 的运行时不直接读取 Hugging Face safetensors。先将 checkpoint 导出到仓库约定位置：

```bash
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

#### 导出 Qwen3

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

从未量化 checkpoint 生成 Fire 自有的 RTN INT4 表示：

```bash
python -B tools/export_qwen3.py \
  --hf models/Qwen3-0.6B \
  --quantization int4 \
  tmp/qwen3-0.6b-int4.fire
```

该模式写 `.fire` v2，量化全部 Linear（含 LM Head），Embedding/Norm 写 FP32。
目前尚未为未量化 8B 实现保留 BF16 Embedding/LM Head 的混合 RTN 导出。

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

三种模式的边界如下，Qwen3 INT4 执行均要求 GPU：

| `--quantization` | Source Checkpoint | 输出布局 |
| --- | --- | --- |
| `none`（默认） | 未量化 BF16 | FP32 `.fire` v1 |
| `int4` | 未量化 BF16 | INT4 投影/LM Head，FP32 Embedding/Norm，`.fire` v2 |
| `awq` | 官方 8B AWQ GEMM | INT4 七类投影，BF16 Embedding/LM Head，FP32 Norm，`.fire` v2 |

所有模式均要求输出路径尚不存在。更多布局与待完成项见
[Qwen3 量化路线](docs/qwen3_quantization_plan.md)。

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

只运行 Matmul/Linear 与 GEMV 的相关正确性检查：

```bash
./build/test/fire_tests \
  --gtest_filter='matmul_test.*:linear_test.*:LinearCudaTest.*'
```

这些检查包含 FP32/BF16、INT4、矩阵回退、长 K、输出行尾、未对齐 Tensor view
以及默认/非默认 CUDA stream。

Python Writer/exporter/quantization contract tests：

```bash
python3 -B test/test_model/test_export_tinyllama.py
python3 -B test/test_model/test_export_qwen3.py
python3 -B test/test_model/test_quant.py
python3 -B test/test_model/test_fire_writer_v2.py
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

8B AWQ 的 GPU smoke test 与独立 HF 对齐使用以下入口；HF 对齐还需要
`transformers`，属于耗时较长的模型验证：

```bash
FIRE_RUN_QWEN3_AWQ_GPU_FORWARD_TEST=1 \
FIRE_QWEN3_AWQ_PATH="$PWD/tmp/qwen3-8B-awq.fire" \
  ./build/test/fire_tests \
  --gtest_filter='Qwen3ModelTest.LocalAwq8BGpuForwardProducesFiniteLogits'

FIRE_RUN_QWEN3_HF_ALIGNMENT_TEST=1 \
  python3 -B test/test_model/test_qwen3_hf_logits_alignment.py \
  --runner build/test/qwen3_logits_runner \
  --hf-model models/Qwen3-8B-AWQ \
  --fire-model tmp/qwen3-8B-awq.fire \
  --awq
```

CTest 中的 `fire_qwen3_hf_logits_alignment` 默认配置为 0.6B FP32；上面的 `--awq`
命令才选择 8B AWQ 参考路径。两 token 对齐、teacher forcing、算子测试与性能复测
是不同范围的验证，新增 GEMV 的正确性测试不代表自动重跑全部 HF/长序列回归。

没有 CUDA device/driver 时，CUDA tests 会跳过。部分真实模型测试还会根据环境变量、
本地模型/tokenizer 文件是否存在和可用显存决定是否跳过；应同时查看 passed 与 skipped。

## CUDA GEMV 与性能记录

当前 prefill 与 decode 都逐 token 调用模型，因此两者的 Linear 输入均为 `N=1`。
CUDA 调度按实际形状和对齐条件选择：

| 权重类型 | 当前单 token 路径 | 其他情况 |
| --- | --- | --- |
| FP32/BF16 | `GEMV_bf16_fp32_v1`；K 为 4 的倍数且地址对齐时向量化加载 | `N=1` 使用标量 GEMV，`N>1` 使用原 tiled dense |
| INT4 | `N=1`、group=128、K 为 128 的倍数且地址对齐时使用 `GEMV_int4_v1` | 使用通用 INT4 kernel |

两条 GEMV 都保持 FP32 激活、累加和输出，并沿用调用方 CUDA stream。当前 8B AWQ
每次 forward 有 252 次 INT4 投影和一次 BF16 LM Head；MHA 使用独立 kernel。
真正的批量 prefill、Tensor Core 路径和跳过中间 prompt logits 尚未实现。

2026-10-21 的 RTX 4070 Laptop GPU 实测，固定 Qwen3-8B-AWQ、160 token 输入
（含模板）、2048 容量、每轮生成 16 token，使用一次预热与三次正式测量的中位数：

| 阶段 | TTFT（ms） | TPOT（ms/token） | Decode（tokens/s） |
| --- | ---: | ---: | ---: |
| 原始 kernel | 44868.69 | 280.23 | 3.57 |
| INT4 GEMV | 5860.49 | 37.33 | 26.78 |
| INT4 + BF16 GEMV | 3742.36 | 23.89 | 41.85 |

最终 decode 吞吐量约为原始基线的 11.72 倍；这是该设备与短负载的记录，不是所有
模型、硬件或上下文长度的性能保证。最新报告的所选 decode 区间中，INT4 GEMV 占
GPU kernel 累计时间的 71.33%，BF16 GEMV 占 22.34%。

`qwen_chat` 每轮打印 TTFT、TPOT 和 Decode throughput。TTFT 不含模型加载；
TPOT/Decode 统计首 token 之后的 forward 与采样等待，不含文本解码和终端输出。
复测应保持提示词、实际生成数量与 `/reset` 方式一致，并在未挂 profiler 时比较
吞吐量；`--profile 8b` 选择模型配置，不会启用 profiler。

详细口径、编译选项和新旧 nsys 数据见 [基线与优化复测](docs/baseline.md)；分析流程见
[性能计划](docs/performance_plan.md)，候选优化点见 [INT4 GEMV 文档](docs/int4_gemv_optimization.md)。
提示词、日志与原始报告放在 Git 忽略的 `profile/`，不会随克隆分发；新增采集应使用
独立报告名以保留可比较的结果。

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
│   ├── quant.py
│   ├── export_tinyllama.py
│   └── export_qwen3.py
├── demo/
│   ├── llama_chat.cpp
│   ├── qwen_chat.cpp
│   └── qwen_chat_utils.h
├── docs/
│   ├── model_export_v0_1.md
│   ├── model_design_v0_1.md
│   ├── qwen3_quantization_plan.md
│   ├── repo_map.md
│   ├── performance_plan.md
│   ├── int4_gemv_optimization.md
│   └── baseline.md
├── CONTEXT.md
├── CMakeLists.txt
└── readme.md
```

核心 C++ 代码编译为静态库 `libfire.a`，并通过 CMake target `Fire::fire` 提供给
测试和 demo。

## 模型执行概览

模型接口每次消费一个 token，prefill 当前也由 demo 逐 token 驱动。已接入的固定
profile 如下；最大 profile 长度不等于已验证的聊天容量或显存承诺：

| Profile | 层数 | Hidden / Q dim | Q / KV heads | 词表大小 | 最大 profile 长度 |
| --- | ---: | --- | --- | ---: | ---: |
| TinyLlama-1.1B | 22 | 2048 / 2048 | 32 / 4 | 32000 | 2048 |
| Qwen3-0.6B | 28 | 1024 / 2048 | 16 / 8 | 151936 | 40960 |
| Qwen3-8B | 36 | 4096 / 4096 | 32 / 8 | 151936 | 40960 |

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

`Qwen3Model` 使用 profile 决定层数和维度，并在 Q/K 投影后增加逐 head RMSNorm。
0.6B 的 Q dim 与 hidden size 不相同；0.6B/8B 共用同一模型实现。当前 AWQ 的
Embedding/LM Head 为 BF16 权重，Norm、激活与 KV Cache 保持 FP32。

`.fire` 是 model-agnostic 的 Tensor Container，不是自描述的通用模型格式。
TinyLlama 的名称、shape 和配置约束由 `TinyllamaLoader` 与固定 Model Profile 负责。

进一步阅读：

- [模型层设计](docs/model_design_v0_1.md)
- [模型导出与 `.fire` v1 格式](docs/model_export_v0_1.md)
- [仓库地图](docs/repo_map.md)
- [Qwen3 量化路线与格式边界](docs/qwen3_quantization_plan.md)
- [CUDA 性能基线与优化结果](docs/baseline.md)
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

### 单 token CUDA 优化：已完成的首轮工作

- [x] 固定 8B AWQ、提示词、上下文与生成长度，记录预热后三次测量的中位数。
- [x] 用 Nsight Systems 的采样边界与调用计数比较 decode kernel 耗时。
- [x] INT4 GEMV：warp 按行、成对解包、向量加载、量化组元数据复用与归约。
- [x] FP32/BF16 GEMV：`N=1` 分流、向量/标量加载、FP32 多累加器与归约。
- [x] 新增边界、未对齐地址和 stream 正确性测试，保留通用回退。
- [x] 记录原始、仅 INT4 GEMV、INT4 + BF16 GEMV 的端到端与 nsys 对比。

### 后续验证与优化

- 在现有固定 token HF 对齐基础上，扩大语料质量回归，并对后续数值改动重新验证
- 用 Nsight Compute 分析代表性热点；扩展长上下文、历史重放、形状与峰值显存测量
- 继续按实测选择 INT4 GEMV、MHA、融合或显存复用的单项优化
- 评估 prefill 仅计算最后 token logits、真正的批量 prefill 与 Tensor Core 路径
- 未量化 8B 的混合 BF16/INT4 RTN 导出和 4096 容量验收
- 更完整的 Loader/状态失败矩阵与 CUDA 错误传播
- temperature、top-k、top-p 等 sampling 策略
- Batch、量化质量回归与更灵活的 Model Profile
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
