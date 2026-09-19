# Fire v0.1 模型层设计

> 设计已确认；实现状态最后核对日期：2026-09-19。Fire v0.1 已发布，TinyLlama FP32 单 token forward、Tokenizer、ArgmaxSampler 与简单聊天生成循环均已接通。

## 1. 范围与当前进度

v0.1 只面向 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 的单序列、单 token、FP32 文本自回归推理。v0.2 接入 Qwen3 时，再根据真实重复代码提取公共实现；Qwen3.5 放在 v0.3。

当前模型核心执行链和 v0.1 CLI 已经闭合：

| 内容 | 当前状态 | 代码入口 |
| --- | --- | --- |
| `ModelConfig`、`Model` | 已定义纯抽象接口，无运行状态成员 | [model.h](../include/Fire/model/model.h) |
| 固定 `TinyLlamaProfile`、结构化权重 | 已定义；默认构造的权重不代表合法模型 | [model_weights.h](../include/Fire/model/model_weights.h) |
| `TinyLlamaBlock` | 已组织两处 Norm 和七个 Linear；由 `TinyLlamaModel` 绑定权重并在模型 forward 中按层调度 | [tinyllama.h](../include/Fire/model/tinyllama.h) |
| `ParamOperator` 移动语义 | 已显式提供移动构造/赋值，继续禁止复制，支持按值存放 Block | [operator.h](../include/Fire/op/operator.h) |
| `FireReader::tensor_count()` | 已实现通用目录数量查询，未打开时为 0 | [fire_reader.h](../include/Fire/model/fire_reader.h) |
| `TinyllamaLoader::open/loader_tensor` | 已实现容器打开与单 Tensor CPU mmap view 查询；不校验完整 profile | [tinyllama_loader.cpp](../src/model/tinyllama_loader.cpp) |
| `TinyllamaLoader::load_weights` | 已校验 201 项 canonical tensor 的数量、名称、FP32 dtype 和 shape；全部成功后发布共享 mmap views | [tinyllama_loader.cpp](../src/model/tinyllama_loader.cpp) |
| Embedding、RoPE、SwiGLU | 已提供 Operator、CPU/CUDA kernel 与数值/错误分支测试 | `include/Fire/op/`、`test/test_op/` |
| Softmax | CPU/CUDA kernel 与直接数值测试已完成；尚无公开 Operator 包装 | [kernels_interface.h](../src/op/kernels/kernels_interface.h) |
| MHA | 已提供公共 Operator、CPU/CUDA GQA kernel、参数校验与数值测试 | [mha.h](../include/Fire/op/mha.h) |
| `TinyLlamaModel`、Runtime、KVCache | 已实现参数绑定、Runtime 分配、紧凑 RoPE cache、K/V 写入与提交、完整 forward 和 reset | 本文第 6、7 节 |
| `LlamaTokenizer` | 已通过 SentencePiece 完成模型加载、BOS/EOS、encode/decode，并覆盖真实 TinyLlama tokenizer 测试 | [llama_tokenizer.cpp](../src/tokenizer/llama_tokenizer.cpp) |
| `ArgmaxSampler` | 已提供 CPU `max_element` 与 CUDA block reduction greedy sampling；相同最大值取首次位置 | [argmax_sampler.cpp](../src/sampler/argmax_sampler.cpp) |
| `llama_chat` | 已实现模型/tokenizer 加载、chat template、greedy 生成、跨轮 KV Cache 复用、固定 2048 token 会话、`/reset` 与 CPU/GPU CLI | [llama_chat.cpp](../demo/llama_chat.cpp) |

文件格式、canonical tensor 名称和导出流程以 [模型导出设计](model_export_v0_1.md) 为准，本文不另定义 wire format。仓库现状见 [仓库地图](repo_map.md)。

## 2. 职责与目标数据流

以下从 `.fire`、文本输入到输出文本的 v0.1 路径均已实现。Tokenizer 和 Sampler 仍保持在 Model 接口之外，由 demo 负责组织生成循环：

```text
.fire 文件
    |
    v
FireReader
    |  容器校验、name -> TensorInfo、shared mmap Buffer
    v
TinyllamaLoader
    |  固定 profile 校验、按名字建立 CPU view
    v
TinyLlamaWeights
    |  可选：显式迁移权重到 GPU，然后绑定为 Parameter
    v
TinyLlamaModel : Model
    +-- Embedding
    +-- TinyLlamaBlock[22]
    |      +-- attention_norm / Wq / Wk / Wv / Wo
    |      +-- ffn_norm / W1 / W2 / W3
    |      +-- RoPE / MHA / SwiGLU / Add
    +-- final Norm / output Linear
    +-- 私有 TinyLlamaRuntime
           +-- 逐层复用的临时 Tensor、RoPE 表
           +-- KVCache：K/V 存储、容量、有效长度

生成循环
    +-- Tokenizer：文本 <-> token IDs
    +-- Model.forward(token_id, pos, logits, context)
    |      +-- Operator -> CPU/CUDA Kernel
    +-- Sampler：logits -> next token
    +-- 停止条件、输出文本
```

职责分工：

- **Tensor/Buffer** 保存数据及其实际存储位置和所有权。
- **OpContext** 描述本次执行的设备、stream、allocator 和临时 workspace，逐次传入，不由 Model 或 Operator 缓存。
- **Operator** 保存计算属性及长期参数；`pos`、当前 KV view 等通过执行参数传入。
- **Loader** 理解具体 Model Profile，输出完整的结构化权重。
- **Model** 绑定参数、安排计算、维护当前序列的具体运行状态，不解析 `.fire`。
- **Sampler/Tokenizer** 属于上层生成流程；Attention 内部 softmax 仍属于模型计算。

`Model` 不继承 `Operator`。当前也不引入通用计算图、Registry、Scheduler、PagedAttention、ModelFactory 或 Transformer/Decoder 继承树。

## 3. Model 公共接口

实际签名位于 `model.h`：

```cpp
const ModelConfig& config() const noexcept;
base::Status prepare(int32_t capacity, const op::OpContext& context);
base::Status forward(int32_t token_id, int32_t pos,
                     tensor::Tensor& logits, const op::OpContext& context);
base::Status reset(const op::OpContext& context);
```

这些方法均为虚接口。公共 `ModelConfig` 只含 `profile_name`、`vocab_size` 和 `max_seq_len`，上层可据此分配 logits 并限制输入长度；hidden size、head 数、FFN 维度等保留在具体 profile。

基类没有数据成员。它统一生命周期与执行结果的含义，不通过增加公共状态来体现价值：

| 项目 | 决定 |
| --- | --- |
| 文件加载、路径、Reader、mmap | 不进入 Model；由 Loader/Reader 负责 |
| `init(path, device)` | 不提供；避免混合加载、绑定、迁移与运行内存准备 |
| device、stream、allocator、workspace 成员 | 不保存；运行参数通过 Context 提供 |
| 通用 KVCache、buffer map | 不进入基类；具体模型持有自己的类型 |
| Sampler、Tokenizer、`predict(..., next)` | 不进入基类；forward 只产生 logits |
| `ModelType` 枚举 | 暂不增加；profile 名用于诊断，不用字符串分派模型实现 |
| `_initialized` | 暂不增加；后续通过成功构建和 Runtime 是否存在区分阶段 |

`profile_name` 使用指向静态字符串的 `string_view`，具体 `config()` 应返回生命周期足够长的不可变配置。TinyLlama 直接使用 `TinyLlamaProfile::model`。

### 生命周期契约（已实现）

1. `TinyLlamaModel::create(weights, context, output)` 接收 Loader 已验证的权重，检查层数、绑定参数，并在 GPU context 下迁移参数。成功后才发布实例，失败保持原输出；它不是通用模型工厂。
2. `prepare(capacity, context)` 要求 `1 <= capacity <= max_seq_len`，并检查 allocator、参数与执行设备一致。在局部 Runtime 中完成准备，成功后替换旧状态并开始空序列。
3. `forward` 要求已成功 prepare，消费一个 token，输出下一 token 的 FP32 logits。
4. `reset(context)` 要求已 prepare，清空序列状态但保留容量、权重及可复用数据。保留 Context 参数，允许具体模型在重置时执行设备操作。

当前 `TinyLlamaModel::create/prepare/forward/reset` 已覆盖上述核心生命周期：forward 校验 token、位置和设备，逐层写入 K/V，生成 logits，并只在全部计算成功后提交 cache length。真实模型双 token 测试会额外验证位置不能重复提交。

一个实例维护一条序列，forward 非 const；暂不允许同一实例被并发调用或交错执行多条序列。Runtime 在类型上独立、所有权上属于具体模型；当前不为外置状态引入 `ModelState` 多态体系。

### Position 与 logits

- `token_id` 在 `[0, vocab_size)`，`pos` 是从 0 开始的绝对位置。
- `pos` 必须等于 KVCache 已接收的有效前缀长度，且小于 capacity。
- 在当前位置写入 K/V 后，Attention 读取 `[0, pos]`，共 `pos + 1` 个位置。
- 所有层及输出投影成功提交后，有效长度才推进一次；提交前发生错误时不推进，输出 logits 不保证有效。
- `pos` 表示调用请求，缓存有效长度表示已接收前缀；二者通过检查保持一致，不在 Model 另存一份 position。
- 调用者预分配 `logits` 为 FP32 `[vocab_size]`，放在执行设备上；Model 不返回内部临时 Tensor 的引用。

v0.1 prompt 按 token 顺序使用相同 forward，最后一个 prompt token 的 logits 才交给 Sampler。Embedding 包含在 Model 中；不要求调用者管理隐藏状态，也无需 `is_prompt` 或单元素 position Tensor。

### 设备与异步执行

不保存 device 并不意味着换 Context 就能换执行设备。权重、运行 Tensor、logits 的实际设备必须与 `context._device_type` 一致；prepare 还要检查 allocator 非空且设备类型一致。只执行预分配算子的 forward 不必强制要求 allocator 存在。

迁移应在参数绑定前完成；之后只改变 Loader 返回的某个 Tensor 句柄，不会自动替换算子已经保存的句柄。不得通过 `Tensor::set_device_type()` 改标签代替实际复制；Buffer 的 device 标签是所有共享 view 共同观察到的属性。

Context 是借用的执行环境。Buffer 为释放存储而持有 allocator 属于所有权信息，不是模型缓存执行环境。`context._workspace` 不能被用于跨调用持有 KV 或返回 logits。

CUDA 成功通常表示已提交，不能据此立即在 CPU 读取结果。同一序列默认沿同一 stream 顺序执行；切换 stream 时由调用者建立依赖。CPU 读取 logits、替换 Runtime 或销毁模型前，调用者必须确认相关设备工作完成。异步设备错误在同步处检查，发生后不能继续把该序列视为有效前缀。

当前设备标识只区分 CPU/GPU，本设计按单 GPU 使用；未来具体设备编号仍属于 Tensor/Context 的职责。

## 4. 固定 profile 与结构化权重

`TinyLlamaProfile` 是 Loader 与后续模型组装共享的 C++ 配置来源，固定为：

```text
hidden_size = 2048             intermediate_size = 5632
num_layers = 22                num_attention_heads = 32
num_kv_heads = 4               head_dim = 64
kv_dim = 256                  kv_groups = 8
vocab_size = 32000             max_seq_len = 2048
rms_norm_eps = 1e-5            rope_theta = 10000
activation = SiLU / SwiGLU     tensor_count = 22 * 9 + 3 = 201
```

head_dim、kv_dim、kv_groups 和 tensor_count 在代码中由固定基础值推导，不另存可独立修改的副本。该推导仅属于 TinyLlama profile，不要求 Qwen3 也满足 residual width 等于 attention projection width。

`TinyLlamaLayerWeights` 每层保存两个 Norm 和七个投影 Tensor；`TinyLlamaWeights` 保存 embedding、22 层权重、final norm 和独立 output。完整 shape 与 canonical name 见 [导出设计第 8 节](model_export_v0_1.md#8-canonical-tensor-mapping)。

v0.1 字段使用 Tensor，因为当前容器和算子只支持 FP32。模型组装时包装为已有 `Parameter` 并交给 `ParamOperator`；成功绑定后不在 Model 另存一套可修改 `_weights`。CPU Tensor 通过共享 mmap Buffer 独立于 Loader/Reader 生命周期持有数据，推理中只读参数。

未来真正支持量化时，将相应权重字段扩展为已有的 `Parameter`，携带 data、QuantConfig、scales 和 zero points。Model 只绑定参数，量化 kernel 选择属于 Operator，不增加模型级量化执行分支。

## 5. Loader 的当前实现

保留现有类名 `TinyllamaLoader` 和 `loader_tensor()`，避免为命名统一改变调用方。`open()` 只验证容器，`loader_tensor()` 只做单名字查询；两者成功都不能证明完整 profile 合法。

`load_weights(TinyLlamaWeights&)` 当前按以下顺序执行：

1. 验证 `tensor_count() == TinyLlamaProfile::tensor_count`；Reader 未打开时数量为 0，因此同样返回 `ModelParseError`。
2. 按固定 canonical name 检查 201 个必需项及 dtype、rank、shape。Reader 已保证名字唯一；总数和所有预期名字一起排除额外 Tensor。
3. 全部 metadata 合法后，在局部 `TinyLlamaWeights` 中用共享 Buffer 和 byte offset 建立 CPU views。
4. 完整组装成功后才移动发布输出；缺项、额外项或 profile 不匹配返回 `ModelParseError`，不会发布局部结果。

Reader 继续只处理通用 wire 不变量、范围和字节数；Loader 不重新读取文件头或手算权重排列。目录顺序应可变化，只要文件符合 v1 的物理排列约束，语义绑定仍按名字完成。

`.fire v1` 不保存模型 metadata。结构验证不能证明文件使用了正确的 eps、theta 或确实来自指定 checkpoint；这些语义依赖固定 profile 的明确选择与受控 exporter。不要将结构符合 profile 描述为来源认证。

当前真实 `.fire` 可用时的集成测试覆盖完整结构化加载、关键 shape 和 Loader 销毁后的 mmap ownership。缺项、额外项、错误 dtype/shape、目录顺序变化以及失败不污染既有输出的独立 fixture 测试仍需补齐。

## 6. Block 与 TinyLlamaModel（已实现）

按 `std::vector<TinyLlamaBlock>` 组织模型，每个 Block 对应一个 `TinyLlamaLayerWeights`。优势是层内参数集中、forward 和绑定关系直接、减少并行 vector 的长度与索引不变量。

AoS 连续存储的是算子对象和 Tensor 句柄，不代表权重数据也连续，更不保证 CUDA kernel 更快。磁盘顺序、C++ 对象组织与 GPU 参数布局应分别决定。Qwen3 到来时可在具体 Block 增加 Q/K Norm，无需修改 TinyLlama 的结构或统一 Norm 索引公式。

`TinyLlamaBlock` 只提供参数成员，不单独暴露 forward；逐层执行顺序由 `TinyLlamaModel::forward` 统一编排。两个 `RmsNormOp` 显式使用 profile 的 `1e-5`，而不依赖通用算子的 `1e-6` 默认值；`ParamOperator` 的显式 move 使 Block 在 vector 扩容时仍能移动已绑定参数。`TinyLlamaModel` 构造函数按层绑定全部 Norm/Linear 参数，并绑定 embedding、final norm 和 output projection。

`TinyLlamaModel` 持有 embedding、RoPE、MHA、Add、SwiGLU、Block vector、final Norm、output Linear 和唯一拥有的 typed Runtime；未 prepare 时 Runtime 为空，基类没有 `_initialized`。当前 forward 已按以下次序执行：

```text
x -> attention_norm -> Wq/Wk/Wv -> Q/K RoPE -> 写 KV -> GQA -> Wo
x + attention_result -> y
y -> ffn_norm -> W1(gate) / W3(up) -> SiLU(gate) * up -> W2(down)
y + ffn_result -> block_output
```

Model 的整体路径为 embedding → 22 个 Block → final Norm → output Linear → logits。位置、层索引和 KV view 通过调用传递，不通过 setter 预先写进 MHA 对象。

### RoPE 布局

Fire exporter 保留 HF Q/K 权重排列，没有 transpose 或 permutation。因此 RoPE 在每个 head 内配对前后半区 `(i, i + head_dim/2)`，sin/cos cache 只保存 `[max_seq_len, head_dim / 2]`。CPU/CUDA 实现和测试已按该布局落地；测试包含非零位置、非对称输入、GQA head 数和非默认 CUDA stream，避免仅以 `pos=0` 掩盖配对错误。[HF Llama 参考实现](https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/llama/modeling_llama.py#L172)

## 7. Typed Runtime 与 KVCache（已实现）

`TinyLlamaRuntime` 使用有名字的 Tensor 字段，不使用 `enum -> map<Tensor>`。`prepare` 创建一套逐层复用的中间 Tensor、紧凑 RoPE cache 和连续 K/V Tensor；后续可在数值基线稳定后继续按算子的别名契约压缩存储。

| 存储 | 当前 shape / 生命周期 |
| --- | --- |
| hidden、block_output、norm_output | `[2048]`；在 Block 间复用 |
| query | `[32, 64]`；当前层临时数据 |
| key、value | 各 `[4, 64]`；当前层临时数据，RoPE 后写入对应 layer/pos 的 cache 槽位 |
| attention_output | `[32, 64]`；MHA 输出，进入 Wo 前 reshape 为 `[2048]` |
| attention_projected、attention_residual | 各 `[2048]`；当前层临时数据 |
| attention_score | `[32, capacity]`；只处理有效前缀 |
| ffn_gate、ffn_up、ffn_activated | 各 `[5632]`；当前层临时数据 |
| ffn_down | `[2048]` |
| rope_cos、rope_sin | `[2048, 32]` 的半区表；prepare 时生成，跨 token、跨 reset 保留 |
| KVCache K/V | 各 `[22, capacity, 4, 64]`；跨 token 保留 |

临时 Tensor 只准备一套并逐层复用，不为 22 个 Block 各分配一套。`KVCache` 封装连续 K/V Tensor、容量和有效长度，支持 CPU 同步写入、GPU stream 异步写入、成功提交与 reset。MHA 接收完整 cache，并通过 `layer_idx` 和 `pos` 只读取当前层的有效前缀；无需额外构造切片 view。`attention_score` 已由 Runtime allocator 分配。

GQA 中每 8 个 query heads 共用一个 KV head，即 `kv_head = query_head / 8`；不把缓存扩展为 32 个 heads。FP32、capacity=2048 时，K/V 合计 `2 * 22 * 2048 * 4 * 64 * 4 = 88 MiB`。

TinyLlama reset 只需把有效长度归零，Attention 必须屏蔽旧内容。长度由整个 token 的执行统一提交，不允许每个 Block 独立推进。

## 8. 验证与后续顺序

当前相关检查：

```bash
cmake -S . -B build
cmake --build build --target fire fire_tests llama_chat -j 4
ctest --test-dir build --output-on-failure
```

`llama_chat` 会加载默认的 `tmp/llama.fire` 和
`models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model`。它按 TinyLlama chat template 显式插入 BOS/EOS，使用 ArgmaxSampler 生成，每轮最多 128 token。system prompt 只在首轮写入，后续轮次只将新增 user/assistant token 追加到同一个 KV Cache；剩余空间不足 128 token 时缩短回复上限并为 assistant EOS 保留一个位置。它不裁剪历史或滑动 KV Cache，达到 2048 token 后结束当前会话。

`test_model.cpp` 覆盖已绑定 Block 在 vector 扩容后的 Linear 数值、两处 Norm 的 TinyLlama epsilon 及抽象/移动属性。Embedding、RoPE、Softmax、SwiGLU 和 MHA 测试覆盖 CPU 数值、错误边界以及可用时的 CUDA 非默认 stream。`test_tinyllama_loader.cpp` 提供显式开启的真实 `.fire` 双 token CPU/GPU forward 测试，验证有限 logits、位置推进、非默认 GPU stream 以及 CPU/GPU logits 最大绝对误差 `< 1e-2`；无 CUDA 或真实模型文件时对应集成测试会跳过。

v0.1 发布后仍需继续补强：

1. **Loader 验收**：补齐缺项、额外项、错误 dtype/shape、目录顺序变化和失败不污染输出的独立 fixture 测试，并完成真实导出数值回读。
2. **状态与错误测试**：现有 KVCache 小尺寸测试已覆盖 K/V 独立写入、提交、reset、错误 shape 与错误位置；继续补齐 `create/prepare/forward/reset` 的失败保持和更多设备错误场景。
3. **参考数值**：对固定 token 序列逐位置比较 Hugging Face FP32 logits；现有 CPU/GPU 一致性测试不能替代独立参考实现。
4. **生成质量与效率**：在现有 greedy CLI 基础上增加随机采样，并评估跨轮 KV Cache 复用；这些能力不属于当前 v0.1 基线。

真实 checkpoint 的全量导入验收仍按导出文档执行。当前真实模型 forward、CPU/GPU 一致性和聊天生成入口已经证明端到端路径可运行；根据 [CONTEXT](../CONTEXT.md) 的严格术语，完整的 Model Support 参考验收仍缺少与 Hugging Face 的独立逐位置 logits/token 对齐记录。

## 9. KuiperLlama 参考与取舍

研究固定于提交 `83030c894da046902c8134110eb70ed59cb49486`。其 `llama3.h/.cpp` 中实际类型名仍是 `LLama2Model/LLama2Layers`。

- 借鉴 Model 组合算子、初始化时绑定参数、预分配临时内存和连续 KV 存储。
- 不沿用基类中的文件读取、Tokenizer/Sampler、device/CUDA 配置和 buffer map；Fire 已有 Reader/Loader 分工与按调用传递的 Context。[Kuiper Model](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/include/model/model.h)
- 不沿用按权重种类分组的多 vector。Qwen3 的 Q/K Norm 需要额外索引公式，共用配置的维度含义也发生变化，因此具体 Block 和具体 profile 更适合 Fire。[Kuiper Qwen3](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/source/model/qwen3.cpp#L187)
- 不复制 legacy offset 解析：其 FP32 `weight(offset)` 按 float 元素数、INT8 按字节偏移。Fire 的模型语义绑定按名字完成，byte offset 只在 Reader/Loader 构造 view 时使用。[RawModelData](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/source/model/raw_model_data.cpp#L16)
