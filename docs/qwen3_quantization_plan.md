# Qwen3 量化路线

状态：量化数学、`.fire` v2 Writer/Reader 和 Qwen3 INT4 Exporter 已有合同测试及小型合成导出验证；Qwen3 Loader、量化 Linear、实际 0.6B/8B 模型验收尚未完成。当前不能把“可导出、可读取”写成量化模型可运行。

## 路线决策

先从未量化的 Qwen3-0.6B 建立并验证 Fire 自有量化与 CPU/CUDA 推理，再把相同路径扩展到 Qwen3-8B，最后导入官方 Qwen3-8B-AWQ。这三个阶段分别验证量化能力、8B 规模扩展和外部格式兼容性；直接接入 AWQ 无法代替 Fire 自己实现量化。量化 payload 使用 `.fire` v2，不改变已发布的 v1 FP32 合同。

官方 AWQ 离线转换为 Fire 的统一量化布局，运行时不增加 AWQ 专属模型或内核；已量化投影不得再次量化，尚未量化的 `lm_head` 可以单独量化以满足 8B 显存目标。Embedding 量化是上述三个阶段之后的独立优化，并非永久排除。

## 统一合同

- 输入分为未量化的 Qwen3 Source Checkpoint 与已量化的官方 AWQ Source Checkpoint。前者由 Fire 离线量化；后者只做格式导入，绝不重新量化已量化投影。两条路径产出相同的 Fire 量化布局。
- 首版是无需校准语料的 RTN、非对称 UInt4 group-wise weight-only；每个输出行沿输入维 K 分组，`group_size = 128`，反量化为 `scale[o,g] × (q[o,k] - zero[o,g])`，其中 `g = floor(k / 128)`。激活和 KV Cache 保持 FP32；不声称 Fire 自己实现了 AWQ 搜索算法。
- 所有 Qwen3 Linear 权重（Q/K/V/O、gate/up/down、`lm_head`）量化；Embedding 和 Norm 保持 FP32。量化 Embedding 是第三阶段之后的独立优化；届时再比较 INT4 与较低浮点精度方案，不预先选定算法。
- 统一物理布局：每个 UInt8 打包两个 UInt4 qweight；每组一个 FP32 scale、一个未打包的 UInt8 zero-point。通用 `Tensor` 只看到字节可寻址的物理 tensor，不引入 0.5 字节元素的 dtype。已锁定偶数列位于低 nibble、奇数列位于高 nibble，量化舍入使用 NumPy `rint` 的 ties-to-even；全零组编码为 `q=0, scale=1, zero=0`。
- `.fire` v2 必须明确保存量化参数关联和 `group_size`；v1 仍只表示 FP32。Qwen3 当前只接受 128，且拒绝 `K % 128 != 0`；Exporter 预检，Loader 防御性复核。模型专用的名称映射、tensor 选择和 profile 校验留在 Qwen3 exporter/loader；量化数学、packing 和 Linear 执行为通用原语。
- `Qwen3Weights` 中可供 Linear 使用的权重统一用 `op::Parameter` 表示：FP32 路径为 `quant_type=None`，INT4 路径携带 qweight、scales、zero-points 和显式 `group_size`。Embedding/Norm 保持普通 Tensor。同一个 `Qwen3Model`、`Qwen3Block`、`LinearOp` 服务两条路径；`Parameter` 的设备迁移、设备校验必须覆盖所有三个量化数据 tensor。
- Exporter 从源 safetensors 分片直接流式写出 v2 量化文件，不先生成约 30.5 GiB 的 FP32 8B `.fire`。初期 CPU 路径重在可读的数值基准；CUDA 路径须避免常驻完整 FP32 反量化权重，并记录速度与峰值显存。

`.fire` v2 保留 32 字节 header 和 96 字节目录项，版本号为 2；wire dtype `1=FP32`、`2=UInt8`。目录项最后 6 字节在 v1 全零，在 v2 编码 `quant_kind:u8, group_size:u32 little-endian, reserved:u8`；普通 tensor 六字节全零，INT4 qweight 使用 `quant_kind=1`、`group_size=128`。量化 Linear 以同一前缀的 `.qweight`、`.scales`、`.zero_points` 三条目录项关联；每条 payload 起点四字节对齐，间隙必须填零。v2 Reader 解析通用元数据，Qwen3 Loader 后续还需复核三件套的名称、shape 和 `group_size=128`。

## 阶段与完成条件

1. **Qwen3-0.6B，建立闭环**：实现 v2 Writer/Reader/Loader、RTN 导出、FP32/INT4 共用的 Linear 参数绑定、CPU 参考和 CUDA 量化 Linear。分别通过 pack/unpack/zero-point 合同、量化 Linear 与浮点反量化参考对齐、CUDA 与 CPU 量化路径对齐、固定 token logits/提示词/语料的端到端质量回归。不能以“能生成文字”代替质量验收。
2. **Qwen3-8B，复用与实机验收**：复用相同量化规则和执行路径，不复制模型类；验证源分片预检、输出回读、端到端单序列 GPU 生成、质量回归、峰值显存和 token 延迟。先尝试 `capacity=4096`；只有实测完整路径因显存不足时才回退 `capacity=2048`，并记录失败条件。即使 2048 也无法运行时，先检查可用显存与临时分配，再重新评估权重/KV 表示；若必须提前改 Embedding 表示，需重新确认阶段顺序，不把它视为自动授权。不得把仅能加载权重视为 Model Support。40,960-token 最大 profile 长度不属于本阶段承诺。
3. **官方 Qwen3-8B-AWQ，导入兼容**：核对官方 checkpoint 的实际 tensor/dtype/packing；将已量化投影无损重排到 Fire 统一布局，将仍未量化的 `lm_head` 单独用 Fire INT4 量化以满足 8B 显存目标，Embedding/Norm 转为现有 FP32 路径。验证每类投影的反量化值与来源参考一致，并完成端到端 logits、质量、显存与生成验收；结果相对官方 AWQ 的差异应单独报告，因为 `lm_head` 在 Fire 侧发生了量化。官方 [配置](https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/main/config.json) 标明 AWQ 4-bit、group size 128、zero-point 和 GEMM；[权重索引](https://huggingface.co/Qwen/Qwen3-8B-AWQ/blob/main/model.safetensors.index.json) 将投影列为 `qweight/qzeros/scales`，`lm_head` 列为普通 `weight`。

质量是阶段门槛，不只是记录项。每阶段在模型级验收前固定测试语料、提示词、参考实现和数值容差；报告 logits 差异、交叉熵或 perplexity 退化及生成样例。若 RTN 达不到门槛，调整量化策略并重新验证，而不放宽“Model Support”的含义。当前没有实测基线，故不预设具体退化百分比。

## 8B 显存预算（估算，非验收结果）

本机 RTX 4070 Laptop GPU 的总显存为 8188 MiB。按本地 Qwen3-8B safetensors 的 shape、所有 Linear（含 `lm_head`）INT4、每组 FP32 scale + UInt8 zero-point、Embedding/Norm FP32 估算，静态权重由 packed qweight **3608.75 MiB**、scales **225.55 MiB**、zero-points **56.39 MiB**、Embedding **2374 MiB**、Norm **1.18 MiB** 构成，合计约 **6265.86 MiB**。现有实现的 FP32 KV Cache 每 token 约 `36 层 × 2(K/V) × 8 KV 头 × 128 × 4 字节 = 0.28125 MiB`；RoPE cache 固定约 20 MiB，Attention score 随 capacity 线性增长。

| Capacity | 静态权重 | FP32 KV Cache | RoPE + score | 最低合计 | 按总显存剩余 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 6265.86 MiB | 576 MiB | 20.25 MiB | 6862.11 MiB | 1325.89 MiB |
| 4096 | 6265.86 MiB | 1152 MiB | 20.50 MiB | 7438.36 MiB | 749.64 MiB |

表中尚未包括其他 Runtime tensor、logits、CUDA context、allocator 额外占用、临时工作区、显存碎片和其他进程，不能据此宣布 4096 已可运行。2026-09-21 查询 `nvidia-smi` 时实际空闲约 **7019 MiB**（已使用约 930 MiB）：在该占用下，4096 的最低合计已比空闲量高约 **419 MiB**；2048 也仅有约 **157 MiB** 的理论余量，可能不足。应在实际可用显存下测量完整初始化、`prepare(capacity)` 和生成的峰值；不清除其他进程或改变既定量化范围来掩盖结果。
