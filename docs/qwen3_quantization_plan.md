# Qwen3 量化路线

状态：量化数学、`.fire` v2 Writer/Reader、Qwen3 INT4 Exporter、Qwen3 Loader 和 GPU INT4 Linear 已有实现及小型合同验证；v2 的 BF16 wire dtype 与 Tensor 字节宽度也已实现。8B 混合 BF16/INT4 导出、Loader 绑定、BF16 Embedding/`lm_head` 算子、官方 AWQ 导入及真实 8B 模型验收尚未完成。格式可读不等于量化模型可运行。

## 路线决策

先从未量化的 Qwen3-0.6B 建立并验证 Fire 自有量化与 CPU/CUDA 推理，再把相同路径扩展到 Qwen3-8B，最后导入官方 Qwen3-8B-AWQ。这三个阶段分别验证量化能力、8B 规模扩展和外部格式兼容性；直接接入 AWQ 无法代替 Fire 自己实现量化。量化 payload 使用 `.fire` v2，不改变已发布的 v1 FP32 合同。

官方 AWQ 离线转换为 Fire 的统一量化布局，运行时不增加 AWQ 专属模型或内核；已量化投影不得再次量化。8B 的 Embedding 与 `lm_head` 保留源 BF16 权重，配合 INT4 投影和 FP32 激活；Embedding 的更低比特量化仍是后续独立优化。

## 统一合同

- 输入分为未量化的 Qwen3 Source Checkpoint 与已量化的官方 AWQ Source Checkpoint。前者由 Fire 离线量化；后者只做格式导入，绝不重新量化已量化投影。两条 8B 路径产出相同的混合 BF16/INT4 Fire 布局。
- 首版是无需校准语料的 RTN、非对称 UInt4 group-wise weight-only；每个输出行沿输入维 K 分组，`group_size = 128`，反量化为 `scale[o,g] × (q[o,k] - zero[o,g])`，其中 `g = floor(k / 128)`。激活和 KV Cache 保持 FP32；不声称 Fire 自己实现了 AWQ 搜索算法。
- 8B 的每层七个投影（Q/K/V/O、gate/up/down）使用 INT4；Embedding 与 `lm_head` 保留 BF16，Norm 保持 FP32。现有 0.6B RTN 基线仍使用 INT4 `lm_head` 和 FP32 Embedding；将 8B 导出路径改为混合布局是后续实现工作。
- 统一物理布局：每个 UInt8 打包两个 UInt4 qweight；每组一个 FP32 scale、一个未打包的 UInt8 zero-point。通用 `Tensor` 只看到字节可寻址的物理 tensor，不引入 0.5 字节元素的 dtype。已锁定偶数列位于低 nibble、奇数列位于高 nibble，量化舍入使用 NumPy `rint` 的 ties-to-even；全零组编码为 `q=0, scale=1, zero=0`。
- `.fire` v2 保存量化参数关联、`group_size` 和可混用的 BF16 普通 tensor；v1 仍只表示 FP32。Qwen3 INT4 当前只接受 group size 128，且拒绝 `K % 128 != 0`；Exporter 预检，Loader 防御性复核。模型专用的名称映射、tensor 选择和 profile 校验留在 Qwen3 exporter/loader；量化数学、packing 和 Linear 执行为通用原语。
- `Qwen3Weights` 中可供 Linear 使用的权重统一用 `op::Parameter` 表示：FP32/BF16 路径为 `quant_type=None`，INT4 路径携带 qweight、scales、zero-points 和显式 `group_size`。Embedding/Norm 保持普通 Tensor。后续 BF16 Embedding 查表输出 FP32，BF16 `lm_head` 接收 FP32 hidden 并输出 FP32 logits；`Parameter` 的设备迁移、设备校验继续覆盖全部量化数据 tensor。
- Exporter 从源 safetensors 分片直接流式写出 v2 量化文件，不先生成约 30.5 GiB 的 FP32 8B `.fire`。初期 CPU 路径重在可读的数值基准；CUDA 路径须避免常驻完整 FP32 反量化权重，并记录速度与峰值显存。

`.fire` v2 保留 32 字节 header 和 96 字节目录项，版本号为 2；wire dtype `1=FP32`、`2=UInt8`、`3=BF16`。BF16 payload 是 little-endian `uint16` 原始位模式，每个元素 2 字节；Python Writer 接收 `np.uint16` 位模式而不做数值转换。目录项最后 6 字节在 v1 全零，在 v2 编码 `quant_kind:u8, group_size:u32 little-endian, reserved:u8`；FP32/BF16 普通 tensor 六字节全零，INT4 qweight 使用 `quant_kind=1`、`group_size=128`。量化 Linear 以同一前缀的 `.qweight`、`.scales`、`.zero_points` 三条目录项关联；每条 payload 起点四字节对齐，间隙必须填零。BF16 只在 v2 有效，不能携带量化元数据；Reader/Writer 已验证这些格式规则。Qwen3 Loader 仍需支持 8B 混合布局并复核完整 profile。

## 阶段与完成条件

1. **Qwen3-0.6B，建立闭环**：实现 v2 Writer/Reader/Loader、RTN 导出、FP32/INT4 共用的 Linear 参数绑定、CPU 参考和 CUDA 量化 Linear。分别通过 pack/unpack/zero-point 合同、量化 Linear 与浮点反量化参考对齐、CUDA 与 CPU 量化路径对齐、固定 token logits/提示词/语料的端到端质量回归。不能以“能生成文字”代替质量验收。
2. **Qwen3-8B，混合 BF16/INT4 与实机验收**：导出每层七个 INT4 投影、BF16 Embedding/`lm_head` 和 FP32 Norm；让 Loader 按名称和 dtype 接受混合布局，并增加 BF16 Embedding 查表及 FP32 输入 × BF16 权重 → FP32 logits 的 Linear 路径。复用同一个 Qwen3Model，验证源分片预检、输出回读、端到端单序列 GPU 生成、质量回归、峰值显存和 token 延迟。先验收 `capacity=2048`，再按实测余量尝试 4096；不足时检查可用显存与临时分配，再决定是否单独调整 KV 表示。不得把仅能加载权重视为 Model Support。40,960-token 最大 profile 长度不属于本阶段承诺。
3. **官方 Qwen3-8B-AWQ，导入兼容**：核对本地 checkpoint 的实际 tensor/dtype/packing；将已量化投影无损重排到 Fire 统一布局，保留 BF16 Embedding 与 `lm_head` 的原始位值，Norm 转为 FP32。验证每类投影的反量化值与来源参考一致，并完成端到端 logits、质量、显存与生成验收。本地配置标明 AWQ 4-bit、group size 128、zero-point 和 GEMM；本地 safetensors 实际包含 I32 `qweight/qzeros`、BF16 `scales`、BF16 Embedding/`lm_head`，导入时应以 tensor metadata 为准。

质量是阶段门槛，不只是记录项。每阶段在模型级验收前固定测试语料、提示词、参考实现和数值容差；报告 logits 差异、交叉熵或 perplexity 退化及生成样例。若 RTN 达不到门槛，调整量化策略并重新验证，而不放宽“Model Support”的含义。当前没有实测基线，故不预设具体退化百分比。

## 8B 显存预算（估算，非验收结果）

本机 RTX 4070 Laptop GPU 总显存为 8188 MiB。按 8B 混合布局估算：每层七个投影 INT4，FP32 scales 与 UInt8 zero-points，Embedding/`lm_head` BF16，Norm、激活和 KV Cache FP32。静态权重包括 packed qweight **3312 MiB**、scales **207 MiB**、zero-points **51.75 MiB**、Embedding **1187 MiB**、`lm_head` **1187 MiB**、Norm **1.18 MiB**，合计约 **5945.93 MiB**。当前 FP32 KV Cache 每 token 为 `36 层 × 2(K/V) × 8 KV 头 × 128 × 4 字节 = 0.28125 MiB`；RoPE cache 固定约 20 MiB，attention score 随 capacity 线性增长。

| Capacity | 静态权重 | FP32 KV Cache | RoPE + score | 最低合计 | 按总显存剩余 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5945.93 MiB | 576 MiB | 20.25 MiB | 6542.18 MiB | 1645.82 MiB |
| 4096 | 5945.93 MiB | 1152 MiB | 20.50 MiB | 7118.43 MiB | 1069.57 MiB |

表中尚未包括其他 Runtime tensor、logits、CUDA context、allocator 额外占用、临时工作区、显存碎片和其他进程。当前只完成 BF16 格式层，表中布局尚不能在模型中加载和执行。后续须在实际可用显存下测量完整初始化、`prepare(capacity)` 和生成峰值；尤其不能据此宣布 4096 已可运行。若 2048 仍不能运行，先检查实际空闲显存与临时分配，再评估 BF16 KV Cache 等独立改造。
