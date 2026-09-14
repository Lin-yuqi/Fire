# Fire v0.1 模型层设计

> 设计已确认；2026-09-14 落地第一、二步的接口与数据结构骨架。本文分别记录当前实现和后续执行契约，不表示 TinyLlama 已能推理。

## 1. 范围与当前进度

v0.1 只面向 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 的单序列、单 token、FP32 文本自回归推理。v0.2 接入 Qwen3 时，再根据真实重复代码提取公共实现；Qwen3.5 放在 v0.3。

本轮对应实施计划前两步的**设计骨架**，完整权重校验算法仍待实现：

| 内容 | 当前状态 | 代码入口 |
| --- | --- | --- |
| `ModelConfig`、`Model` | 已定义纯抽象接口，无运行状态成员 | [model.h](../include/Fire/model/model.h) |
| 固定 `TinyLlamaProfile`、结构化权重 | 已定义；默认构造的权重不代表合法模型 | [model_weights.h](../include/Fire/model/model_weights.h) |
| `TinyLlamaBlock` | 已组织两处 Norm 和七个 Linear；未绑定权重、无 block forward | [tinyllama.h](../include/Fire/model/tinyllama.h) |
| `ParamOperator` 移动语义 | 已显式提供移动构造/赋值，继续禁止复制，支持按值存放 Block | [operator.h](../include/Fire/op/operator.h) |
| `FireReader::tensor_count()` | 已实现通用目录数量查询，未打开时为 0 | [fire_reader.h](../include/Fire/model/fire_reader.h) |
| `TinyllamaLoader::open/loader_tensor` | 已实现容器打开与单 Tensor CPU mmap view 查询；不校验完整 profile | [tinyllama_loader.cpp](../src/model/tinyllama_loader.cpp) |
| `TinyllamaLoader::load_weights` | 已声明并提供占位实现；返回 `FunctionUnImplement`，不修改输出 | [tinyllama_loader.h](../include/Fire/model/tinyllama_loader.h) |
| `TinyLlamaModel`、Runtime、KVCache | 尚未实现；下文给出后续组织方式 | 本文第 6、7 节 |

文件格式、canonical tensor 名称和导出流程以 [模型导出设计](model_export_v0_1.md) 为准，本文不另定义 wire format。仓库现状见 [仓库地图](repo_map.md)。

## 2. 职责与目标数据流

以下是完整目标路径，其中 Model 执行与 Runtime 部分尚待实现：

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

### 生命周期契约（待具体模型实现）

1. 具体模型的 `create(weights, output)` 验证内存中的权重结构并绑定参数。成功后才发布实例，失败保持原输出；它不是通用模型工厂。
2. `prepare(capacity, context)` 要求 `1 <= capacity <= max_seq_len`。在局部 Runtime 中完成准备，成功后替换旧状态并开始空序列；失败保留旧 Runtime。它不迁移权重。
3. `forward` 要求已成功 prepare，消费一个 token，输出下一 token 的 FP32 logits。
4. `reset(context)` 要求已 prepare，清空序列状态但保留容量、权重及可复用数据。保留 Context 参数，允许具体模型在重置时执行设备操作。

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

## 5. Loader 的后续实现入口

保留现有类名 `TinyllamaLoader` 和 `loader_tensor()`，避免为命名统一改变调用方。`open()` 只验证容器，`loader_tensor()` 只做单名字查询；两者成功都不能证明完整 profile 合法。

`load_weights(TinyLlamaWeights&)` 的占位实现始终返回未实现状态。下一步依次完成：

1. 检查 Reader 已打开；验证 `tensor_count() == TinyLlamaProfile::tensor_count`。
2. 按固定 canonical name 检查 201 个必需项及 dtype、rank、shape。Reader 已保证名字唯一；总数和所有预期名字一起排除额外 Tensor。
3. 全部 metadata 合法后，在局部 `TinyLlamaWeights` 中用共享 Buffer 和 byte offset 建立 CPU views。
4. 完整组装成功后才发布输出；失败保持输出原状。缺项、额外项或 profile 不匹配返回 `ModelParseError`，不把外部输入错误交给 `CHECK/LOG(FATAL)`。

Reader 继续只处理通用 wire 不变量、范围和字节数；Loader 不重新读取文件头或手算权重排列。目录顺序应可变化，只要文件符合 v1 的物理排列约束，语义绑定仍按名字完成。

`.fire v1` 不保存模型 metadata。结构验证不能证明文件使用了正确的 eps、theta 或确实来自指定 checkpoint；这些语义依赖固定 profile 的明确选择与受控 exporter。不要将结构符合 profile 描述为来源认证。

## 6. Block 与完整 TinyLlamaModel（后续）

按 `std::vector<TinyLlamaBlock>` 组织模型，每个 Block 对应一个 `TinyLlamaLayerWeights`。优势是层内参数集中、forward 和绑定关系直接、减少并行 vector 的长度与索引不变量。

AoS 连续存储的是算子对象和 Tensor 句柄，不代表权重数据也连续，更不保证 CUDA kernel 更快。磁盘顺序、C++ 对象组织与 GPU 参数布局应分别决定。Qwen3 到来时可在具体 Block 增加 Q/K Norm，无需修改 TinyLlama 的结构或统一 Norm 索引公式。

当前 `TinyLlamaBlock` 只提供参数成员。两个 `RmsNormOp` 显式使用 profile 的 `1e-5`，而不依赖通用算子的 `1e-6` 默认值；`ParamOperator` 的显式 move 使 Block 在 vector 扩容时仍能移动已绑定参数。

后续 `TinyLlamaModel` 的私有成员计划为 embedding 算子、Block vector、final Norm、output Linear 以及唯一拥有的 typed Runtime。未成功 prepare 时 Runtime 为空；基类不增加 `_initialized`。RoPE/MHA/SwiGLU/Add 接入 Block 后，按以下计算次序执行：

```text
x -> attention_norm -> Wq/Wk/Wv -> Q/K RoPE -> 写 KV -> GQA -> Wo
x + attention_result -> y
y -> ffn_norm -> W1(gate) / W3(up) -> SiLU(gate) * up -> W2(down)
y + ffn_result -> block_output
```

Model 的整体路径为 embedding → 22 个 Block → final Norm → output Linear → logits。位置、层索引和 KV view 通过调用传递，不通过 setter 预先写进 MHA 对象。

### RoPE 布局

Fire exporter 保留 HF Q/K 权重排列，没有 transpose 或 permutation。因此 RoPE 必须在每个 head 内配对前后半区 `(i, i + head_dim/2)`，不能直接复制采用相邻维度配对的实现。仅验证 `pos=0` 无法揭示该错误，测试必须包含非零位置和非对称输入。[HF Llama 参考实现](https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/llama/modeling_llama.py#L172)

## 7. Typed Runtime 与 KVCache（后续）

使用有名字的 Tensor 字段，不使用 `enum -> map<Tensor>`。先保留独立中间结果，数值基线通过后再按算子的别名契约复用存储。

| 存储 | 建议 shape / 生命周期 |
| --- | --- |
| hidden、block_output、norm_output | `[2048]`；在 Block 间复用 |
| query | `[2048]`；当前层临时数据 |
| key、value | 各 `[256]`；初版先独立存储，后续可直接写 cache 槽位 |
| attention_output、attention_projected、attention_residual | 各 `[2048]`；当前层临时数据 |
| attention_score | `[32, capacity]`；只处理有效前缀 |
| ffn_gate、ffn_up、ffn_activated | 各 `[5632]`；当前层临时数据 |
| ffn_down | `[2048]` |
| rope_cos、rope_sin | 可保存 `[capacity, 32]` 的半区表；跨 token、跨 reset 保留 |
| KVCache K/V | 各 `[22, capacity, 4, 64]`；跨 token 保留 |

临时 Tensor 只分配一套，逐层复用，不为 22 个 Block 各分配一套。KVCache 独立封装存储、容量、有效长度和槽位/历史访问；Tensor views 应共享 backing Buffer，不产生缺少所有权的悬空指针。

GQA 中每 8 个 query heads 共用一个 KV head，即 `kv_head = query_head / 8`；不把缓存扩展为 32 个 heads。FP32、capacity=2048 时，K/V 合计 `2 * 22 * 2048 * 4 * 64 * 4 = 88 MiB`。

TinyLlama reset 只需把有效长度归零，Attention 必须屏蔽旧内容。长度由整个 token 的执行统一提交，不允许每个 Block 独立推进。

## 8. 验证与后续顺序

当前相关检查：

```bash
cmake -S . -B build
cmake --build build -j 4
ctest --test-dir build --output-on-failure -R '^(ModelStructureTest\.|FireReaderTest\.|op_test\.|linear_test\.|matmul_test\.|rmsnorm_test\.)'
```

`test_model.cpp` 覆盖已绑定 Block 在 vector 扩容后的 Linear 数值，以及两处 Norm 的 TinyLlama epsilon；同时在编译期检查抽象接口和 Block 的复制/移动属性。Reader 的现有合同测试增加未打开/已打开目录数量检查。这些测试不代表完整 Loader、模型组装或 forward 已完成。命令还包含既有 `op_test.add` CUDA 测试，无 CUDA 环境时会跳过。

后续依次实现并验收：

1. **完整 Loader**：缺项、额外项、错误 shape、目录顺序变化、失败不污染输出、Reader 销毁后 view 有效；核心测试不依赖 GPU 或真实权重。完整 shape 可以使用稀疏文件和少量哨兵数据。
2. **KVCache/Runtime**：小容量验证跨层跨位置读写、越界拒绝、长度提交和 reset 隔离。
3. **Embedding/RoPE/SwiGLU**：小 Tensor 对照独立公式，特别验证非零位置的 HF RoPE 布局。
4. **MHA/Block/Model**：先验证 GQA 与有效历史范围，再核对一个真实 Block 的中间结果，最后接通 final norm/head。
5. **数值与生成**：先给定相同 token 序列逐位置比较 FP32 logits，再验证 CUDA 与 CPU，最后接 greedy sampler 和生成循环。

真实 checkpoint 的全量导入验收仍按导出文档执行。单个权重的 RMSNorm 测试、结构骨架或文件存在都不能替代 Export Compatibility；Model Support 还要求完整端到端生成与参考校验。

## 9. KuiperLlama 参考与取舍

研究固定于提交 `83030c894da046902c8134110eb70ed59cb49486`。其 `llama3.h/.cpp` 中实际类型名仍是 `LLama2Model/LLama2Layers`。

- 借鉴 Model 组合算子、初始化时绑定参数、预分配临时内存和连续 KV 存储。
- 不沿用基类中的文件读取、Tokenizer/Sampler、device/CUDA 配置和 buffer map；Fire 已有 Reader/Loader 分工与按调用传递的 Context。[Kuiper Model](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/include/model/model.h)
- 不沿用按权重种类分组的多 vector。Qwen3 的 Q/K Norm 需要额外索引公式，共用配置的维度含义也发生变化，因此具体 Block 和具体 profile 更适合 Fire。[Kuiper Qwen3](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/source/model/qwen3.cpp#L187)
- 不复制 legacy offset 解析：其 FP32 `weight(offset)` 按 float 元素数、INT8 按字节偏移。Fire 的模型语义绑定按名字完成，byte offset 只在 Reader/Loader 构造 view 时使用。[RawModelData](https://github.com/zjhellofss/KuiperLLama/blob/83030c894da046902c8134110eb70ed59cb49486/kuiper/source/model/raw_model_data.cpp#L16)
