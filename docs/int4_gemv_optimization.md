# INT4 GEMV 可选优化点

本文针对 `matmul_quant_kernel_cu` 的单向量计算路径，区分已经接入的优化与后续候选点。状态核对日期：2026-10-23，测量结果见 [baseline.md](baseline.md)。每次根据测量结果选择一项，确认正确性和收益后，再决定下一项；不预设版本数量或要求逐项实现。

## 当前计算与实现

逻辑计算是 `X[N,K] × W[M,K]ᵀ → Y[N,M]`。当 `N=1` 时，它就是 GEMV：

```text
y[m] = Σ x[k] × scale[m,g] × (q[m,k] - zero[m,g])
g = k / group_size
```

当前量化布局：

| 数据 | 类型与物理形状 |
| --- | --- |
| input | FP32，`[K]` 或 `[N,K]` |
| qweight | UInt8，`[M,K/2]`；偶数 k 放低 4 bit，奇数 k 放高 4 bit |
| scales | FP32，`[M,K/group_size]` |
| zero_points | UInt8，`[M,K/group_size]`，每组一个字节 |
| output | FP32，`[M]` 或 `[N,M]` |

当前已有两条路径：

- 快速路径 `GEMV_int4_v1`：`N=1`、group_size=128、K 为 128 的倍数，输入满足 16 字节、packed 权重满足 2 字节对齐。每个 warp 处理一行，每个 block 四个 warp；每 lane 使用 `uint16_t` 和 `float4` 读取四个权重与输入，复用组内 scale/zero，FP32 累加后进行 warp 归约。
- 通用回退 `matmul_quant_int4_fp32_naive`：当前使用 `block(16,16)`，每线程计算一个输出并串行遍历 K，覆盖 `N>1`、其他合法 group_size 或未对齐的输入视图。

`matmul_quant_kernel_cu` 是 host 调度函数，已经按上述条件分派；两条路径保持原有量化布局、FP32 激活与输出，并使用调用方传入的 stream。

Fire 当前 prefill 也逐 token 调用 forward，因此 `N==1` 的优化会影响当前 prefill 和 decode。

## 优化点概览

前四类数据组织优化已合并进入当前实现，具体结果见基线文档；其余条目是独立候选，不表示均已实现。

| 优化点 | 想减少的成本 | 当前状态与适用条件 |
| --- | --- | --- |
| 一个 warp 协作计算一行 | 单线程的长累加链、跨行分散读取 | 已实现；需要 warp 归约，求和顺序改变 |
| 一次解包一个 byte 中的两个 INT4 | 重复的字节读取、索引和解包 | 已实现；当前每 lane 连续处理两个 byte |
| 按 group 组织循环，复用 scale/zero | 每个元素重复计算组索引、读取元数据 | 已实现 group=128 专用路径 |
| 向量化读取权重和输入 | 加载指令数量、访存请求效率 | 已实现 `uint16_t` / `float4`；需满足对齐条件 |
| 调整每个 block 的 warp 数 | block 数量或资源分配不合适 | 当前固定四个 warp，其他参数尚待比较 |
| 多个累加器、适度循环展开 | 累加依赖和循环控制开销 | INT4 尚未引入多个累加器；需观察寄存器与 spill |
| 在 block 内复用输入 tile | 多个输出行重复加载 x | 候选；比较 L1 缓存复用与 shared memory 同步成本 |
| 多个 warp 协作计算一行 | 大 K 下单个 warp 的串行工作 | 候选；增加跨 warp 归约成本 |
| 有限的形状或 group_size 特化 | 通用路径中的除法、分支和不合适参数 | 目前只有 group=128 的快速路径，更多特化待测 |

## 已实现：一个 warp 计算一行

让 32 个线程沿 K 维分担一行权重的点积。例如，一个 block 放 4 个 warp：

```text
block = 128 threads
每个 block 计算 4 个输出
grid = ceil(M / 4)

lane = threadIdx.x % 32
warp = threadIdx.x / 32
m = blockIdx.x * 4 + warp

每个 group 的起点 base = group * 128
lane 0 处理 k = base + 0..3
lane 1 处理 k = base + 4..7
...
lane 31 处理 k = base + 124..127

各 lane 累加自己的部分和
warp 归约后，由 lane 0 写 output[m]
```

这样既增加 block 数量，也让同一 warp 沿一行读取相邻数据。注意 grid 的分母是每个 block 的输出行数 4，而不是线程数 128。

本次同时接入 warp 分工、成对解包、量化组复用和向量加载，保留原有量化公式与 FP32 累加。当前实测体现这组改动的整体收益，没有逐项消融数据。

边界处理需要保证：无效输出行对应的整个 warp 不参与计算；有效行中没有元素可处理的 lane 使用零部分和，并继续参与归约。使用 `__shfl_down_sync` 时，参与线程必须满足 mask 和调用一致性的要求。

## 成对解包与量化组复用

当前一个 byte 保存两个权重，可以一次取出低、高两个 nibble，并与两个输入元素相乘。权重继续在寄存器中反量化，直接参与 FP32 累加。

按当前 Qwen 的 group_size=128，一组包含 64 个 packed byte。一个 warp 可以让每个 lane 处理两个 byte，共四个权重：

```text
组号：g
每个 lane 的组内 byte 位置：2*lane、2*lane + 1
byte 位置 p 对应的输入位置：g*128 + 2*p，以及后一项
这一组使用相同的 scale[m,g] 和 zero[m,g]
```

当前每个 lane 每组读取一次 scale/zero，并复用于四个权重。lane 0 加载元数据再广播仍是候选；同址读取本身可能被硬件合并，不能预设手工广播一定更快。

通用接口允许其他合法 group_size。若使用 128 的专用实现，需要保留其他情况的 fallback；尤其奇数组大小时，同一 byte 中的两个权重可能属于不同组。

## 已实现：向量化加载

当前快速路径一次读取多个 packed byte，以及匹配的连续输入元素。

实际实现为每个 lane 读取两个 packed byte（一个 `uint16_t`）和四个 FP32 输入（一个 `float4`）。32 个 lane 正好覆盖一组 128 个权重，并使用这组的元数据。

需要先确认：

- packed 权重地址满足 2 字节对齐，输入地址满足 `float4` 的 16 字节对齐；不能只根据分配基地址推断 Tensor view 的实际对齐。
- 行跨度、组边界和尾部处理正确；不满足条件时走通用 INT4 kernel。
- 编译器是否已经合并了相邻加载。手写向量类型不保证生成更好的指令。

直接使用每 lane 一个 `uint32_t` 会覆盖八个权重，整 warp 覆盖 256 个权重，跨越两个 128 元素量化组，不能继续假设全 warp 共用一份元数据。

## 其他候选如何决定

- **block 参数**：例如比较每个 block 4 个或 8 个 warp，保持每 warp 一行。结合 block 数量、寄存器、实际 occupancy 和 kernel 时间判断。
- **多累加器与展开**：在单线程仍有较长依赖链时尝试。检查寄存器和 spill，避免以减少循环指令换来更多显存访问。
- **输入 tile 复用**：同一 block 的多个输出行都使用 x，可以考虑协作加载。先判断 L1 是否已有效复用，再评估 shared memory 的收益及同步成本。
- **多 warp 一行**：例如将大 K 的行拆给多个 warp，再归约部分和。它可能适合较长的点积，但不一定优于每 warp 一行。
- **Split-K**：只有 M 较小、K 很大且并行度仍不足时，再考虑跨 block 拆分。额外归约或原子操作会增加流量、同步和数值行为的复杂度。
- **Tensor Core**：需要单独设计激活精度、量化计算方式和 tile，不能直接将当前 W4A32 点积替换成一条指令。可在负载和测量支持时再评估。

这些候选无需全部尝试。Nsight Systems 用于判断时间分布；NCU 的硬件指标和实际输入形状用于帮助解释原因。低 occupancy 或低 DRAM 吞吐都不能单独作为瓶颈结论。

## 比较与验证

`timeline_base.nsys-rep` 对应原始 16×16 INT4 kernel，`timeline_gemmv.nsys-rep` 对应 INT4 GEMV，`timeline_gemmv_v2.nsys-rep` 还包含普通 BF16 输出投影的 GEMV。三个阶段的无 profiler Decode 中位数分别为 3.57、26.78、41.85 tokens/s；硬件、输入和计时口径见 [baseline.md](baseline.md)。后续应与当前 INT4 + BF16 GEMV 阶段比较，不能把两个阶段的合计收益都归给 INT4。

同一文件中的 `GEMV_bf16_fp32_v1` 已为普通 FP32/BF16 权重提供 `N=1` 快速路径；其四个独立累加器不代表 INT4 kernel 也已采用该策略。

Qwen3-8B 的代表性投影形状如下，均为 N=1：

| 投影 | M | K |
| --- | ---: | ---: |
| Q / O | 4096 | 4096 |
| K / V | 1024 | 4096 |
| gate / up | 12288 | 4096 |
| down | 4096 | 12288 |

每次只选一个主要变化，并检查：

1. 固定输入下与独立 FP32 参考的误差。并行归约改变求和顺序，不要求逐 bit 一致，但需要合理的绝对和相对误差标准。
2. M 的尾部、`[K]` 与 `[1,K]` 输入、N>1 fallback、非默认 stream，以及专用路径以外的合法 group_size。
3. 代表形状的 kernel 时间，以及相同 prompt、上下文起点和生成数量下的无 profiler demo 指标。

现有线性层测试已经覆盖两条 GEMV 的多种 K、`[K]` / `[1,K]`、默认/非默认 stream、输出行尾和未对齐回退，并覆盖 INT4 的其他合法 group_size。后续实施某项优化时再补对应的必要覆盖。测量命令、版本标识和结果追加到现有 [baseline.md](baseline.md)，不为每个候选点预先生成脚本或报告。

## 相关位置与参考

- [CUDA matmul 实现](../src/op/kernels/cuda/matmul_kernel.cu)
- [量化线性层验证与分派](../src/op/linear.cpp)
- [现有 matmul / linear 测试](../test/test_op/test_matmul.cpp)
- [整体性能分析计划](performance_plan.md)
- [CUDA SIMT 与访存合并](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [CUDA shuffle 等语言扩展](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
