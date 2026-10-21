# INT4 GEMV 可选优化点

本文针对 `matmul_quant_kernel_cu` 的单向量计算路径，列出可以独立尝试的优化点。每次根据测量结果选择一项，确认正确性和收益后，再决定下一项；不预设版本数量或要求逐项实现。

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

当前代码采用一维 `block(256)`，每个线程计算一个输出并串行遍历 K。相邻线程访问不同权重行；例如 M=1024 时只有 4 个 block。

`matmul_quant_kernel_cu` 是 host 调度函数。新增 GEMV 路径可以在其中按 `N==1` 分派，保持 `N>1` 的通用路径，以及原有量化布局、FP32 激活和 FP32 累加。GPU kernel 需要声明为 `__global__`，并使用传入的 stream 启动。

Fire 当前 prefill 也逐 token 调用 forward，因此 `N==1` 的优化会影响当前 prefill 和 decode。

## 优化点概览

优先级表示当前值得研究的顺序，具体收益需要测量。

| 优化点 | 想减少的成本 | 适用条件与代价 |
| --- | --- | --- |
| 一个 warp 协作计算一行 | 单线程的长累加链、跨行分散读取 | 需要 warp 归约；求和顺序改变 |
| 调整每个 block 的 warp 数 | block 数量或资源分配不合适 | 应与具体 M、K 一起比较，线程多不保证更快 |
| 一次解包一个 byte 中的两个 INT4 | 重复的字节读取、索引和解包 | 两个元素可能跨量化组，需正确求元数据索引 |
| 按 group 组织循环，复用 scale/zero | 每个元素重复计算组索引、读取元数据 | 当前 Qwen 的 group=128 可做专用路径 |
| 向量化读取权重和输入 | 加载指令数量、访存请求效率 | 需要满足对齐、边界和组划分条件 |
| 多个累加器、适度循环展开 | 累加依赖和循环控制开销 | 可能增加寄存器使用和 spill |
| 在 block 内复用输入 tile | 多个输出行重复加载 x | 需比较 L1 缓存复用与 shared memory 同步成本 |
| 多个 warp 协作计算一行 | 大 K 下单个 warp 的串行工作 | 增加跨 warp 归约成本 |
| 有限的形状或 group_size 特化 | 通用路径中的除法、分支和不合适参数 | 增加代码与分派维护成本，需保留 fallback |

## 最值得先研究：一个 warp 计算一行

让 32 个线程沿 K 维分担一行权重的点积。例如，一个 block 放 4 个 warp：

```text
block = 128 threads
每个 block 计算 4 个输出
grid = ceil(M / 4)

lane = threadIdx.x % 32
warp = threadIdx.x / 32
m = blockIdx.x * 4 + warp

lane 0 处理 k = 0, 32, 64, ...
lane 1 处理 k = 1, 33, 65, ...
...
lane 31 处理 k = 31, 63, 95, ...

各 lane 累加自己的部分和
warp 归约后，由 lane 0 写 output[m]
```

这样既增加 block 数量，也让同一 warp 沿一行读取相邻数据。注意 grid 的分母是每个 block 的输出行数 4，而不是线程数 128。

首个实验只改变线程分工，保留逐元素解包、原有量化公式和 FP32 累加，方便判断收益来自哪里。

边界处理需要保证：无效输出行对应的整个 warp 不参与计算；有效行中没有元素可处理的 lane 使用零部分和，并继续参与归约。使用 `__shfl_down_sync` 时，参与线程必须满足 mask 和调用一致性的要求。

## 成对解包与量化组复用

当前一个 byte 保存两个权重，可以一次取出低、高两个 nibble，并与两个输入元素相乘。权重继续在寄存器中反量化，直接参与 FP32 累加。

按当前 Qwen 的 group_size=128，一组包含 64 个 packed byte。一个 warp 可以让每个 lane 处理两个 byte，共四个权重：

```text
组号：g
每个 lane 的组内 byte 位置：lane、lane + 32
byte 位置 p 对应的输入位置：g*128 + 2*p，以及后一项
这一组使用相同的 scale[m,g] 和 zero[m,g]
```

先单独比较成对解包，再尝试将元数据读取移到组循环外层，避免同时改变 block 参数。lane 0 加载元数据再广播也是候选，但同址读取本身可能被硬件合并，不能预设手工广播一定更快。

通用接口允许其他合法 group_size。若使用 128 的专用实现，需要保留其他情况的 fallback；尤其奇数组大小时，同一 byte 中的两个权重可能属于不同组。

## 向量化加载

在 warp 分工和组处理方式稳定后，可以研究一次读取多个 packed byte，以及匹配的连续输入元素。

一个与 group=128 匹配的候选是：每个 lane 读取两个 packed byte（一个 `uint16_t`）和四个 FP32 输入（一个 `float4`）。32 个 lane 正好覆盖一组 128 个权重，并使用这组的元数据。

需要先确认：

- packed 权重地址满足 2 字节对齐，输入地址满足 `float4` 的 16 字节对齐；不能只根据分配基地址推断 Tensor view 的实际对齐。
- 行跨度、组边界和尾部处理正确；不满足条件时走标量路径。
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

已有 `timeline_base.nsys-rep` 对应原先的 16×16 block，当前源码已改为一维 256 线程。先分别标明这两个实现，不能将旧报告当作当前一维版本的实测结果。下一项优化应明确与哪个实现比较。

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

复用现有线性层测试作为起点，实施某项优化时再补对应的必要覆盖。测量命令、版本标识和结果可以追加到现有 `profile/baseline.md`，不为每个候选点预先生成脚本或报告。

## 相关位置与参考

- [CUDA matmul 实现](../src/op/kernels/cuda/matmul_kernel.cu)
- [量化线性层验证与分派](../src/op/linear.cpp)
- [现有 matmul / linear 测试](../test/test_op/test_matmul.cpp)
- [整体性能分析计划](performance_plan.md)
- [CUDA SIMT 与访存合并](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [CUDA shuffle 等语言扩展](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
