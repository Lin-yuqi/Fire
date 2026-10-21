# Qwen3-8B-AWQ Release 基线与优化复测

原始基线测量开始时间：2026-10-21 18:55:09（Asia/Shanghai）。

## 输入与环境

| 项目 | 本次设置 |
| --- | --- |
| 程序 | `build/demo/qwen_chat` |
| 模型 | `tmp/qwen3-8B-awq.fire` |
| Tokenizer | `models/Qwen3-8B-AWQ` |
| GPU / 驱动 | RTX 4070 Laptop GPU / 610.62 |
| 构建 | Release，CUDA 架构 `sm_89` |
| 编译选项 | C++：`-O3 -DNDEBUG -g`；CUDA：`-O3 -DNDEBUG -lineinfo` |
| GPU 频率 | 默认动态频率 |
| 输入 | 当前单行版本的 `profile/prompt.txt` |
| Prompt 长度 | 包含聊天模板共 160 token |
| 会话 | 单序列，关闭 thinking，每轮从空 KV 会话开始 |
| 上下文容量 | 2048 token |
| 生成上限 / 实际生成 | 每轮 16 / 16 token，均触及上限 |
| 重复方式 | 同一进程内 1 轮预热、3 轮正式测量，每轮结束后 `/reset` |

输入文件 SHA-256：

```text
d56776e94dcfc172e6111c2ff48f676f601dd6cfcb612458c1d88334057cceec
```

输入处理方式：去除各非空行的首尾空白，用一个空格连接；当前文件只有一行。测量结束时确认有效输入未改变。

本次使用的程序参数：

```bash
./build/demo/qwen_chat --profile 8b --context-size 2048 --max-new-tokens 16
```

标准输入依次发送四次相同提示词，每次后接 `/reset`，最后发送 `/exit`。没有挂载 profiler。

## 测量结果

以下数值直接取自 demo 输出，保留其两位小数精度。

| 轮次 | TTFT（ms） | TPOT（ms/token） | Decode（tokens/s） |
| --- | ---: | ---: | ---: |
| 预热，不计入汇总 | 44831.37 | 283.17 | 3.53 |
| 正式 1 | 44868.69 | 280.23 | 3.57 |
| 正式 2 | 44846.92 | 281.11 | 3.56 |
| 正式 3 | 44876.64 | 279.58 | 3.58 |
| **正式测量中位数** | **44868.69** | **280.23** | **3.57** |
| 正式测量范围 | 44846.92–44876.64 | 279.58–281.11 | 3.56–3.58 |

四轮生成文本一致。每轮结束时缓存均为 `178/2048`，与 `160 个输入 token + 16 个生成 token + 2 个收尾 token` 一致；各轮没有复用上一轮的会话前缀。

整个进程墙钟时间为 **203.43 秒**，包含模型加载、四轮请求、文本输出、收尾及退出；模型加载时间未单独测量。

## 如何理解这些数字

- **TTFT 约 44.87 秒**：从 prompt 编码前到首个 token 采样完成，包含分词、160 token 的逐 token prefill 和首次采样。模型加载发生在该计时之外；TTFT 也不是单独的 prefill 耗时。
- **TPOT 约 280.23 ms/token**：累计一次生成 token 的 forward 到下一次采样返回 CPU 的时间，包括采样及其等待。16 个生成 token 中，首个计入 TTFT，后续 15 个计入 decode 指标。
- **Decode 约 3.57 tokens/s**：由上述 15 个 token 的累计时间计算。每步文本解码和终端输出被排除，因此它不等于用户实际收到文本的速度。
- 达到生成上限时，最后一个生成 token 的 forward，以及两个收尾 token 的 forward，仍会执行，但不计入这 15 次 decode 的累计时间。阅读后续时间线时要排除这部分。

本次三个正式样本的 decode 范围为 3.56–3.58 tokens/s，可作为这个固定负载后续复测的参照。具体 kernel 的耗时占比留到下一步分析。

## GEMV_int4_v1 优化后复测（2026-10-21）

本节数据摘录自 INT4 优化阶段的 `profile/after.log`；该文件随后被新一轮复测覆盖，历史数值保留在本节。原测量未挂载 profiler，沿用上述 Release 编译参数、模型、提示词、上下文容量和生成上限；同一进程运行一次预热、三次正式测量，每轮 `/reset`。当时核对 `prompt.txt` 的 SHA-256 与原始基线一致。没有记录该轮独立的测量开始时间或 GPU 频率。

本次改动为 INT4 的 `N=1`、`group_size=128` 且满足对齐要求的 GEMV 路径：每个 warp 处理一行，使用 `float4` / `uint16_t` 加载、成对解包、组内 scale/zero 复用和 warp 归约。BF16 dense 路径未修改。

| 轮次 | TTFT（ms） | TPOT（ms/token） | Decode（tokens/s） |
| --- | ---: | ---: | ---: |
| 预热，不计入汇总 | 5858.67 | 36.90 | 27.10 |
| 正式 1 | 5834.90 | 37.27 | 26.83 |
| 正式 2 | 5862.13 | 37.36 | 26.77 |
| 正式 3 | 5860.49 | 37.33 | 26.78 |
| **正式测量中位数** | **5860.49** | **37.33** | **26.78** |
| 正式测量范围 | 5834.90–5862.13 | 37.27–37.36 | 26.77–26.83 |

四轮均触及 16 token 上限，生成文本在本轮四次请求之间一致；每轮结束的缓存仍为 `178/2048`。

| 指标 | 原始基线中位数 | 优化后中位数 | 对比 |
| --- | ---: | ---: | --- |
| TTFT | 44868.69 ms | 5860.49 ms | 耗时减少 86.94%，加速约 7.66 倍 |
| TPOT | 280.23 ms/token | 37.33 ms/token | 耗时减少 86.68%，加速约 7.51 倍 |
| Decode | 3.57 tokens/s | 26.78 tokens/s | 吞吐量约为原来的 7.50 倍 |

加速倍数按各指标显示的两位小数计算；TPOT 与 Decode 的比值存在舍入差异。正式轮次的吞吐量范围为 26.77–26.83 tokens/s，本次收益明显超过这些样本的轮次间波动。

## INT4 优化阶段的 Nsight Systems 报告对比

报告为 `profile/timeline_base.nsys-rep` 和 `profile/timeline_gemmv.nsys-rep`。分析时仅将 SQLite 临时导出到 `/tmp`，没有在 `profile/` 中额外保存统计文件。这里不使用 profiler 内的吞吐量替代上面的无 profiler 复测。

两个报告各有 32 次 `argmax_kernel_fp32`，对应两轮各 16 次采样。分别选取第二轮第一次采样 kernel 结束至第 16 次采样 kernel 结束的区间：其中恰有 15 次模型 forward，对应 15 次 dense、3780 次 INT4 和 540 次 MHA，排除了 prefill、最后生成 token 的 forward 及收尾 forward。边界来自 GPU 采样 kernel，未包含两端完整的 CPU 采样处理，因此不是 demo TPOT 计时区间的严格复刻。

| 区间 | 原始报告 | 优化后报告 |
| --- | ---: | ---: |
| 报告内起止时间 | 98.605603655–102.858346542 s | 20.144032299–20.703861647 s |
| 区间墙钟时间 | 4252.743 ms | 559.829 ms |
| GPU kernel 累计时间 | 4194.854 ms | 533.507 ms |

| Kernel | 区间调用次数（两份报告相同） | 原始累计耗时（ms） | 优化后累计耗时（ms） | 优化后占比 |
| --- | ---: | ---: | ---: | ---: |
| INT4：naive → `GEMV_int4_v1` | 3780 | 3927.281 | 243.885 | 45.71% |
| `matmul_kernel_cu_dense<16, BF16>` | 15 | 247.439 | 268.070 | 50.25% |
| `mha_kernel_cu_fp32` | 540 | 10.082 | 10.804 | 2.03% |

占比的分母为该区间所有 GPU kernel 的累计执行时间，不能直接当作端到端耗时占比。全进程汇总中，优化后 dense 占比为 51.07%、INT4 GEMV 为 46.03%；这组数包含预热、prefill 和收尾，与上表的区间统计不同。

INT4 的平均调用耗时从 1038.96 µs 降至 64.52 µs，约加速 16.10 倍；这一阶段最大的单项热点转为 BF16 dense。dense 平均每次从 16.50 ms 变为 17.87 ms，源码没有随本次 INT4 优化修改；未锁定频率和运行状态，不能单凭两次采集将这部分差异归因于代码退化。

这一阶段 AWQ 模型的 dense 对应 `lm_head` / `output.weight`，输入 FP32 `[4096]`，权重 BF16 `[151936,4096]`，输出 FP32 `[151936]`，即 `N=1,M=151936,K=4096`。已核对本地 `.fire` 文件的张量目录，权重字节数为 1,244,659,712，约 1.16 GiB。prefill 也逐 token 执行，每个 token 都调用一次这条 dense 路径。

该阶段提出的下一项改动为 `N=1` 的 BF16 GEMV，现已实现，结果记录在下一节。

## GEMV_bf16_fp32_v1 优化后复测（2026-10-21）

数据来源为当前 `profile/after.log`，未挂载 profiler。仍使用同一模型、提示词、2048 上下文、16 token 生成上限、关闭 thinking，以及一次预热、三次正式测量、每轮 `/reset` 的方式。再次核对 `prompt.txt` 的 SHA-256 与原始基线一致，`build/CMakeCache.txt` 中 Release、CUDA 架构 89 及上述编译参数保持一致。日志未记录本轮独立的测量开始时间或 GPU 频率。

本轮在保留 INT4 GEMV 的基础上，为普通权重增加 `N=1` 的 GEMV 分派：warp 按行处理，每 block 四个 warp；FP32 输入和权重使用 `float4` 加载，BF16 权重使用 `uint2` 加载后转 FP32，使用四个独立累加器与 warp shuffle 归约。K 或地址不满足向量化条件时使用标量 GEMV，`N>1` 保留原 dense 路径。本次 AWQ 模型的输出投影使用 BF16 向量化分支，结果不代表 FP32 权重分支的独立性能测量。

| 轮次 | TTFT（ms） | TPOT（ms/token） | Decode（tokens/s） |
| --- | ---: | ---: | ---: |
| 预热，不计入汇总 | 3757.65 | 24.16 | 41.38 |
| 正式 1 | 3742.36 | 23.89 | 41.85 |
| 正式 2 | 3723.19 | 23.88 | 41.87 |
| 正式 3 | 3759.70 | 24.09 | 41.52 |
| **正式测量中位数** | **3742.36** | **23.89** | **41.85** |
| 正式测量范围 | 3723.19–3759.70 | 23.88–24.09 | 41.52–41.87 |

四轮均生成 16 token，每轮结束的缓存仍为 `178/2048`。本轮四次请求的生成文本一致，也与上一轮日志中记录的可见文本一致；未单独记录 token ID 或逐元素 logits。

| 指标 | 原始基线中位数 | 仅 INT4 GEMV 中位数 | INT4 + BF16 GEMV 中位数 | 相比上一轮 |
| --- | ---: | ---: | ---: | --- |
| TTFT | 44868.69 ms | 5860.49 ms | 3742.36 ms | 耗时减少 36.14%，加速约 1.57 倍 |
| TPOT | 280.23 ms/token | 37.33 ms/token | 23.89 ms/token | 耗时减少 36.00%，加速约 1.56 倍 |
| Decode | 3.57 tokens/s | 26.78 tokens/s | 41.85 tokens/s | 吞吐量提高 56.27%，约为上一轮的 1.56 倍 |

相对原始基线，TTFT 约加速 11.99 倍，TPOT 约加速 11.73 倍，Decode 吞吐量约为原来的 11.72 倍。以上均按显示精度计算；本轮正式样本吞吐量范围为 41.52–41.87 tokens/s，收益明显超过这些样本的轮次间波动。

本轮 nsys 报告为 `profile/timeline_gemmv_v2.nsys-rep`，与上一阶段的 `profile/timeline_gemmv.nsys-rep` 对比。继续仅在 `/tmp` 临时导出 SQLite，分析后删除。报告仍有 32 次采样，选择第二轮第一次采样 kernel 结束至第 16 次采样 kernel 结束，保持上一节的 GPU 边界口径。

| 区间 | 仅 INT4 GEMV 报告 | INT4 + BF16 GEMV 报告 |
| --- | ---: | ---: |
| 报告内起止时间 | 20.144032299–20.703861647 s | 11.617647910–11.983810962 s |
| 区间墙钟时间 | 559.829 ms | 366.163 ms |
| GPU kernel 累计时间 | 533.507 ms | 341.042 ms |

| Kernel | 区间调用次数（两份报告相同） | 上一轮累计耗时（ms） | 本轮累计耗时（ms） | 本轮占比 |
| --- | ---: | ---: | ---: | ---: |
| `GEMV_int4_v1` | 3780 | 243.885 | 243.275 | 71.33% |
| BF16 输出投影：dense → `GEMV_bf16_fp32_v1<BF16, true>` | 15 | 268.070 | 76.188 | 22.34% |
| `mha_kernel_cu_fp32` | 540 | 10.804 | 10.819 | 3.17% |

本轮每个输出投影平均约 5.08 ms，上一轮为 17.87 ms，约加速 3.52 倍；INT4 的平均耗时从 64.52 µs 变为 64.36 µs，基本不变。报告中的调用次数仍对应 15 次完整 forward，且输出投影已进入 BF16 向量化 GEMV 分支。

上表占比的分母为该 decode 区间的 GPU kernel 累计时间。全进程汇总中，INT4 GEMV 占 72.66%、BF16 GEMV 占 22.76%，包含预热、prefill 和收尾，不能混用。当前主要 GPU 耗时重新集中在 INT4 GEMV；这里仅更新测量结果，不自动开始下一项优化。

## Nsight Systems 采集命令参考

在仓库根目录运行下列命令。输入方式与基线相同，运行两轮：第一轮预热，第二轮用于分析。以下为通用命令模板；实际新旧报告名称见上一节，后续采集应使用独立名称以保留对比报告。关闭自动统计导出，不额外生成统计文件。

```bash
python3 - <<'PY' | nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --stats=false \
  --output=profile/timeline \
  ./build/demo/qwen_chat --profile 8b --context-size 2048 --max-new-tokens 16
from pathlib import Path
prompt = ' '.join(
    line.strip() for line in Path('profile/prompt.txt').read_text().splitlines()
    if line.strip()
)
for _ in range(2):
    print(prompt)
    print('/reset')
print('/exit')
PY
```

打开报告后，先定位第二轮的首个 `argmax_kernel_fp32`。它之前是该轮 prefill；它之后到第 16 次采样之间，包含 15 次已计时的 decode。全进程汇总混入了加载、预热、prefill 和收尾，不能直接作为 decode 占比。`nsys` 运行中打印的吞吐量也不替代上述无 profiler 基线。
