# Fire 模型导出与 `.fire` v1 格式

> 状态：Accepted。Q1-Q15 的格式、职责、失败语义和验收边界已经确认。

> 实现进度（2026-09-20）：Fire v0.1 已发布。FireWriter、TinyLlama exporter、FireReader、`TinyllamaLoader` 与 core wire/profile 合同测试均已完成并接入 CTest。Qwen3 exporter 已支持按 `config.json` 自动识别 0.6B/8B profile，并读取单文件或 HF shard index；0.6B 的 311 项完整 FP32 文件已通过 Loader、两 token CPU forward 和 Hugging Face logits 对齐，8B 的 399 项/5 分片真实 checkpoint metadata preflight 也已通过。Qwen3 GPU forward 和量化 wire 尚不在已完成范围内。模型接口与执行设计见 [模型层设计](model_design_v0_1.md)。

## 1. 目标

Fire v0.1 只为 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 建立一条极简模型导入路径：

```text
Hugging Face checkpoint
        -> TinyLlama-specific Python exporter
        -> .fire
        -> C++ FireReader (mmap)
        -> TinyLlama ModelLoader
        -> TinyLlamaWeights
        -> TinyLlamaModel / Operator（完整 FP32 forward 已实现）
```

`.fire` 是可 mmap 的 Fire Tensor Container，而不是通用或自描述的模型 checkpoint。它通过 tensor directory 消除 exporter 与 loader 对隐式 tensor 排列顺序的依赖。

当前开发分支还建立了 Qwen3 的导出路径：

```text
Qwen3-0.6B / Qwen3-8B local checkpoint
        -> profile-selected Qwen3 exporter
        -> FP32 .fire v1
        -> FireReader / Qwen3Loader / Qwen3Model（0.6B CPU 已验证）
```

## 2. 能力与版本边界

- Fire v0.1 绑定 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`。
- Fire v0.2 首先绑定 `Qwen/Qwen3-0.6B`；0.6B/8B 共用一个由 profile value 驱动的 exporter 和 Loader，不复制两套模型类。
- Qwen3.5 独立放到 Fire v0.3，不反向扩大 v0.1/v0.2 的抽象。
- `.fire` Format Version 独立于 Fire Project Version；本文定义的首版 wire value 为 `1`。
- `.fire` v1 wire/payload 只支持 64-bit little-endian host 上的 FP32、rank 1/2 tensor；TinyLlama/Qwen3 的 BF16 source tensor 由 exporter 转成 FP32，不输出 BF16、FP16 或量化 payload。
- Export Compatibility 的验收是 C++ 解析全部 tensor，并对选定 tensor 做数值回读；Model Support 仍要求端到端生成和参考实现校验。

TinyLlama exporter 的输入边界仍是本地、精确匹配该 Model Profile 的 `config.json` 和单个 `model.safetensors`。Qwen3 exporter 额外支持标准 `model.safetensors.index.json` 和其引用的本地分片。两个 exporter 都不从 Hugging Face Hub 下载文件，不执行 remote code，也不实例化 Transformers model 或完整 `state_dict`。

## 3. 职责

### TinyLlama exporter

- 验证本地 HF `config.json` 与固定 TinyLlama profile 完全匹配。
- 验证源 safetensors 的 201 个 tensor 名、dtype 和 shape。
- 将 HF tensor name/layout 适配为 Fire canonical name/layout。
- 将 BF16 转为 FP32，并保证 tensor 为 C-contiguous。
- 用内部两遍算法协调 directory 规划与逐 tensor payload 写出。

### Qwen3 exporter

- 根据 model-semantic config 字段自动选择 0.6B 或 8B profile，不要求调用方重复指定型号。
- 从 profile 动态推导 canonical descriptor；0.6B 的 `q_dim = 2048`，独立于 `hidden_size = 1024`。
- 同时支持单个 `model.safetensors` 和 HF shard index，并严格核对 index、实际 shard keys、BF16 dtype、shape 与总字节数。
- 将同一 shard 的目录项排在一起，使 metadata/payload 两遍中每个 shard 都只打开一次；Loader 仍只按 canonical name 查找，不依赖目录顺序。
- 与 TinyLlama 相同，在创建输出前完成 metadata preflight，并逐 tensor 转换、写出 FP32 payload。

### FireWriter

- 只认识已经规范化的 Fire tensor 名、dtype、shape、byte size 和 payload。
- 编码 Header、Tensor Directory 与 raw data。
- 不理解 Hugging Face、safetensors、TinyLlama、Qwen3 或模型配置。
- 不暴露 public planner、provider、streaming policy 或 WriteReport。

### FireReader

- 使用 `open + fstat + mmap` 打开整个 `.fire` 文件。
- 验证 Header、Tensor Directory 和 payload 范围。
- 建立 `name -> TensorInfo` 索引；`find(name)` 在未打开或名称不存在时均返回 `nullptr`。
- 通过 `tensor_count()` 返回目录项数；尚未成功打开时为 `0`，具体模型应有多少项由 Loader 判断。
- 通过 `mapped_buffer()` 提供共享 mmap backing storage，使 Tensor 可以独立于 Reader 生命周期持有 mapping。
- 不创建模型、Layer 或 Operator，也不理解具体 Model Profile。

### TinyLlama ModelLoader

- 与模型组装共用 `TinyLlamaProfile` 的精确配置，按 canonical name 读取 tensor；不解析 HF/safetensors。
- 完整加载入口严格验证 201 个 canonical tensor 的数量、名称、FP32 dtype 和 shape，再交付 `TinyLlamaWeights`；byte size 已由 FireReader 的通用 wire 校验保证。
- 用 FireReader 的共享 mmap Buffer 和绝对 byte offset 构造 CPU Tensor view。
- 权重迁移是绑定前的显式步骤；具体 Model 负责将结构化权重绑定到 Operator。

当前 `open()` 只验证 `.fire` 容器；`loader_tensor()` 只读取单个具名 Tensor，均不能证明 TinyLlama profile 完整有效。`load_weights()` 通过总数和全部预期 canonical name/dtype/shape 验证固定 profile，在局部对象完整组装后才发布共享 mmap views；系统性的缺项、额外项和错误 metadata fixture 测试仍待补齐。

## 4. `.fire` Format Version 1

所有整数和 FP32 payload 使用 little-endian。wire 字段按下表的 byte offset 显式编码和解码，禁止直接序列化或 `reinterpret_cast` native C++ struct。

### 4.1 Header

Header 固定为 32 bytes：

| Byte offset | Wire type | Field | v1 constraint |
| ---: | --- | --- | --- |
| 0 | `char[8]` | magic | `FIRECKPT` |
| 8 | `uint32` | format_version | `1` |
| 12 | `uint32` | tensor_count | TinyLlama 为 `201`；Qwen3-0.6B/8B 为 `311`/`399` |
| 16 | `uint64` | tensor_directory_offset | v1 为 `32` |
| 24 | `uint64` | data_offset | directory 结束后的绝对 byte offset |

file size 由 `fstat` 获得，不在 Header 重复保存。

### 4.2 TensorInfo

每项固定为 96 bytes：

| Byte offset | Wire type | Field | v1 constraint |
| ---: | --- | --- | --- |
| 0 | `char[64]` | name | 非空、NUL-padded ASCII，最多 63 bytes |
| 64 | `uint64` | byte_offset | 相对文件起点的绝对 byte offset |
| 72 | `uint64` | byte_size | `product(shape) * 4` |
| 80 | `uint32[2]` | shape | 未使用维度为 `0` |
| 88 | `uint8` | dtype | v1：`1 = FP32` |
| 89 | `uint8` | ndim | v1：`1` 或 `2` |
| 90 | `byte[6]` | padding | 必须全为 `0` |

wire dtype 编号不复用 C++ `base::DataType` 的枚举序号；FireReader 负责显式映射。

### 4.3 Raw data

- 第一项 payload 从 Header 的 `data_offset` 开始。
- payload 按 Tensor Directory 顺序紧密相连，不插入 tensor 间 padding。
- 所有 byte offset 至少满足 FP32 的 4-byte alignment。
- 目录顺序不携带模型语义；ModelLoader 只按 canonical name 查找。
- TinyLlama 的 directory 占 `201 * 96 = 19,296` bytes，因此 `data_offset = 19,328`。
- TinyLlama FP32 payload 为 `4,400,193,536` bytes，完整文件预期为 `4,400,212,864` bytes。
- Qwen3-0.6B 的 directory 占 `311 * 96 = 29,856` bytes，因此 `data_offset = 29,888`；FP32 payload 为 `3,006,529,536` bytes，完整文件预期为 `3,006,559,424` bytes。
- Qwen3-8B 的 directory 占 `399 * 96 = 38,304` bytes，因此 `data_offset = 38,336`；FP32 payload 为 `32,762,941,440` bytes，完整文件预期为 `32,762,979,776` bytes。该 FP32 文件约 30.5 GiB，只用于当前格式兼容，不是最终 8B 量化交付形态。

## 5. Reader 必须验证的不变量

- host 必须能够安全表示 wire `uint64` offset/size 并直接解释 little-endian FP32；v0.1 只支持 64-bit little-endian host。
- `tensor_directory_offset` 必须严格等于 `32`。
- magic 必须逐 byte 等于 `FIRECKPT`，`format_version` 必须等于 `1`。
- `tensor_count * 96` 必须使用 checked arithmetic，且 `data_offset` 必须严格等于 `32 + tensor_count * 96`，不得越过文件末尾。
- tensor name 必须非空、在 64 bytes 内存在 NUL、仅含 ASCII 且全局唯一；首个 NUL 之后的 name bytes 必须全为零。
- 每项 TensorInfo 的 6 个 padding bytes 必须全为零。
- dtype、ndim、shape 必须属于 v1 范围，shape 必须可转换为当前 Tensor 的 `int32` dims。
- `product(shape) * 4` 必须使用 checked arithmetic，且等于 `byte_size`。
- 每个 payload 必须位于 data region 内、至少 4-byte aligned，并与上一项紧密相连。
- 最后一个 payload 的结束位置必须等于 `fstat` 文件大小。

## 6. mmap 与 Tensor 生命周期

- FireReader 使用 `open(O_RDONLY)` 和 `mmap(PROT_READ, MAP_PRIVATE)`；mmap 成功后可以关闭 fd。
- Reader 内部使用一个私有 `MMapBuffer : base::Buffer` 包装整个 mapping，析构时调用 `munmap`。
- Reader 和由 ModelLoader 构造的 CPU Tensor 共享该 Buffer；最后一个持有者析构后才解除 mapping。
- FireReader 提供最小 ownership seam：`std::shared_ptr<base::Buffer> mapped_buffer() const`。
- Reader 未打开时，`mapped_buffer()` 返回空 `shared_ptr`。
- ModelLoader 使用 `Tensor(dtype, dims, shared_buffer, absolute_byte_offset)`，不使用只借用裸指针的 `Tensor::from_blob()`。
- mmap 权重是不可变数据；通用 readonly Tensor 类型推迟。
- CUDA v0.1 使用同步 H2D copy；pinned staging 和异步加载推迟。
- 任一 Reader 或 mapped Tensor 存活期间，不得原地 truncate/overwrite 对应 `.fire` 文件。

## 7. TinyLlama 固定模型配置

`.fire` v1 不保存模型 metadata。C++ 固定配置定义于 `include/Fire/model/model_weights.h` 的 `TinyLlamaProfile`，由 Loader 与模型组装共用：

| Field | Value |
| --- | ---: |
| hidden size | 2048 |
| intermediate size | 5632 |
| layers | 22 |
| attention heads | 32 |
| KV heads | 4 |
| head dim | 64 |
| vocabulary size | 32000 |
| RMS epsilon | 1e-5 |
| RoPE theta | 10000 |
| max positions | 2048 |
| activation | SiLU / SwiGLU |
| tied embeddings | false |

因此 v0.1 必须显式调用 TinyLlama ModelLoader，不提供由 `.fire` 自动选择模型的 generic `load_model(path)`。

## 8. Canonical tensor mapping

| HF source pattern | Fire canonical pattern | Shape |
| --- | --- | --- |
| `model.embed_tokens.weight` | `tok_embeddings.weight` | `[32000, 2048]` |
| `model.layers.{i}.input_layernorm.weight` | `layers.{i}.attention_norm.weight` | `[2048]` |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attention.wq.weight` | `[2048, 2048]` |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attention.wk.weight` | `[256, 2048]` |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attention.wv.weight` | `[256, 2048]` |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` | `[2048, 2048]` |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.ffn_norm.weight` | `[2048]` |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers.{i}.feed_forward.w1.weight` | `[5632, 2048]` |
| `model.layers.{i}.mlp.down_proj.weight` | `layers.{i}.feed_forward.w2.weight` | `[2048, 5632]` |
| `model.layers.{i}.mlp.up_proj.weight` | `layers.{i}.feed_forward.w3.weight` | `[5632, 2048]` |
| `model.norm.weight` | `norm.weight` | `[2048]` |
| `lm_head.weight` | `output.weight` | `[32000, 2048]` |

`{i}` 取 `0..21`。所有二维 tensor 保持 C-contiguous `[out_features, in_features]`，不 transpose、fuse、permute 或 kernel-pack。源 checkpoint 的 embedding 与 output 未绑定，两者都必须导出。

### 8.1 Qwen3 profile 与 canonical mapping

Qwen3 exporter 目前识别两个 dense profile：

| Field | Qwen3-0.6B | Qwen3-8B |
| --- | ---: | ---: |
| hidden size | 1024 | 4096 |
| intermediate size | 3072 | 12288 |
| layers | 28 | 36 |
| attention heads | 16 | 32 |
| KV heads | 8 | 8 |
| head dim | 128 | 128 |
| q dim | 2048 | 4096 |
| KV dim | 1024 | 1024 |
| vocabulary size | 151936 | 151936 |
| tied embeddings config | true | false |
| canonical tensors | 311 | 399 |

两个 profile 共用以下映射，`{i}` 的范围由 profile 的 layer count 决定：

| HF source pattern | Fire canonical pattern | Shape |
| --- | --- | --- |
| `model.embed_tokens.weight` | `tok_embeddings.weight` | `[vocab_size, hidden_size]` |
| `model.layers.{i}.input_layernorm.weight` | `layers.{i}.attention_norm.weight` | `[hidden_size]` |
| `model.layers.{i}.self_attn.q_proj.weight` | `layers.{i}.attention.wq.weight` | `[q_dim, hidden_size]` |
| `model.layers.{i}.self_attn.k_proj.weight` | `layers.{i}.attention.wk.weight` | `[kv_dim, hidden_size]` |
| `model.layers.{i}.self_attn.v_proj.weight` | `layers.{i}.attention.wv.weight` | `[kv_dim, hidden_size]` |
| `model.layers.{i}.self_attn.o_proj.weight` | `layers.{i}.attention.wo.weight` | `[hidden_size, q_dim]` |
| `model.layers.{i}.self_attn.q_norm.weight` | `layers.{i}.attention.q_norm.weight` | `[head_dim]` |
| `model.layers.{i}.self_attn.k_norm.weight` | `layers.{i}.attention.k_norm.weight` | `[head_dim]` |
| `model.layers.{i}.post_attention_layernorm.weight` | `layers.{i}.ffn_norm.weight` | `[hidden_size]` |
| `model.layers.{i}.mlp.gate_proj.weight` | `layers.{i}.feed_forward.w1.weight` | `[intermediate_size, hidden_size]` |
| `model.layers.{i}.mlp.down_proj.weight` | `layers.{i}.feed_forward.w2.weight` | `[hidden_size, intermediate_size]` |
| `model.layers.{i}.mlp.up_proj.weight` | `layers.{i}.feed_forward.w3.weight` | `[intermediate_size, hidden_size]` |
| `model.norm.weight` | `norm.weight` | `[hidden_size]` |
| `lm_head.weight` | `output.weight` | `[vocab_size, hidden_size]` |

0.6B 的 source checkpoint 即使配置为 tied embeddings，也实际包含 `lm_head.weight`；当前 `.fire` profile 因而显式保存 embedding 与 output 两项。导出命令为：

```bash
python -B tools/export_qwen3.py \
  --hf models/Qwen3-0.6B \
  tmp/qwen3-0.6b.fire
```

exporter 会从 `config.json` 自动选择 profile。输出路径必须不存在。当前真实 0.6B checkpoint 已导出约 2.80 GiB FP32 payload，并用于 Loader、CPU forward 和 Hugging Face logits 对齐；完整导出本身不属于自动化测试。8B 也能通过相同 CLI 读取标准分片 checkpoint，但 Fire v1 只会生成约 30.5 GiB 的 FP32 文件，并不提供 INT4/INT8 编码。

## 9. 内部两遍导出

导出前先完成 config、source key、dtype、shape、canonical name 长度和唯一性检查。随后：

1. Metadata pass：通过 `safe_open(source_path, framework="numpy")` 的 slice metadata 读取 name、dtype 和 shape，不构造 payload tensor；建立私有的 source-to-canonical 描述列表，并 checked 计算全部 absolute byte offset。完整 preflight 成功后 exclusive-create 目标文件，再写出 Header 与 Tensor Directory。单文件 checkpoint 只打开一次；Qwen3 分片 checkpoint 每个 shard 打开一次。
2. Payload pass：通过 `safe_open(source_path, framework="pt", device="cpu")` 按目录顺序逐项 `get_tensor()`，转换成 FP32 C-contiguous tensor，通过共享内存的 NumPy view 写入 payload，然后立即释放当前临时 tensor。单文件 checkpoint 使用一个长期 context；Qwen3 会把同 shard 项分组，因此每个 shard 也只打开一次。

该列表和两遍算法是 exporter/writer 的 implementation detail，不成为 public TensorSource、provider 或 planning interface。

最小 internal seam 是：model-specific adapter 持有私有 `ExportEntry(source_name, shard_path, TensorInfo)` 列表并驱动 safetensors 循环；FireWriter 只接收 Fire `TensorInfo[]`、写出 Header/Directory，并按相同顺序接收单个规范化 tensor。FireWriter 不接触 source name、shard、safetensors handle 或 HF config。

Python 实现应避免两个不必要的内存放大点：

- 不用 `AutoModelForCausalLM` 或完整 `state_dict` 加载模型；metadata pass 使用 `safe_open(..., framework="numpy")` 与 `get_slice()`。
- payload 将 little-endian FP32 NumPy view 转为 `memoryview(fp32).cast("B")` 并直接交给目标文件的 `write()`，不用会复制整个 tensor 的 `.tobytes()`；v0.1 exporter 在非 little-endian host 上直接拒绝运行。

对当前 TinyLlama 的本地验证结果：metadata pass 峰值 RSS 约 27 MiB。payload pass 在 Python 对象层面只同时持有当前 tensor，算法层最大约为 125 MiB BF16 输入加 250 MiB FP32 输出；使用单个长期存在的 safetensors context 时，已触碰的 mmap 页面仍可能累计计入进程 RSS，本地实测峰值约 2.6 GiB。v0.1 接受该代价，以避免重复打开 safetensors，同时仍避免完整 4.4 GB FP32 模型作为 tensor 对象常驻。

不得预先把输出文件 `ftruncate` 到最终大小：否则中途失败可能留下 file size 合法但 payload 为零的伪完整文件。源 safetensors 在两遍导出期间不得被修改；每项 payload 写出时仍需重新核对实际 dtype、shape、byte size 和最终文件位置。

## 10. 明确不做

- checksum、compression、`.fire` 输出分片。Qwen3 只支持读取 HF source shards，输出仍是一个连续 `.fire` 文件。
- streaming writer policy/framework。
- atomic commit、replace/version migration。
- WriteReport、public planning interface、lazy-loading policy framework。
- schema/model registry、自动 ModelLoader 选择。
- FireReader 复用或重新 `open` 另一文件的状态语义；v0.1 每个文件使用新的 Reader。
- universal quantization、LoRA merge、GGUF 或 PyTorch pickle。
- 通用 Hugging Face 架构支持、MoE/视觉/MTP 预抽象；Qwen3 仅支持显式列出的 0.6B/8B dense profile。
- tokenizer/chat template 打包。
- transpose、QKV fusion 或设备专用 packing。

## 11. 失败语义

### Python exporter

- 全部 metadata preflight 在创建输出文件前完成；失败时不得触碰目标路径。
- 目标使用 `open(path, "xb", buffering=0)` exclusive-create 新建；路径已存在时直接失败，不 truncate 或替换。
- 只有 Header、Directory、全部 payload、最终位置检查和文件关闭均成功后才保留目标。创建目标后的任意普通异常或 `Ctrl-C` 均先关闭文件，再 best-effort 删除本次创建的 partial file；删除失败只追加 warning，不覆盖原始错误。
- `SIGKILL` 或断电仍可能留下 partial file。未预分配、顺序写出的短文件会被结构/长度校验拒绝；但 v0.1 没有 crash durability 或 payload corruption detection 保证，缺少 checksum/fsync 时无法保证发现长度恰好正确但内容损坏的文件。
- 不提供 `--force`，不使用临时文件加 rename，也不实现 fsync、resume、rollback 或 transaction。
- 错误信息至少包含失败阶段、相关路径或 canonical/source tensor name，以及可用的 expected/actual 值。

### FireReader

外部文件错误必须返回 `base::Status`，不得触发 `CHECK` 或 `LOG(FATAL)`：

| Failure | Status |
| --- | --- |
| 空路径、文件不存在、权限不足或 `open` 失败 | `PathNotValid` |
| `fstat`、`mmap` 等系统操作失败 | `InternalError` |
| 任一 Header、Directory 或 payload invariant 失败 | `ModelParseError` |
| 成功 | `Success` |

`open()` 应先在局部 RAII 状态中完成 fd、mapping、解析和索引构建，全部成功后才提交到 Reader 成员。失败后 Reader 保持未打开状态。

```cpp
base::Status FireReader::open(const std::string& path);
const TensorInfo* FireReader::find(std::string_view name) const;
size_t FireReader::tensor_count() const noexcept;
std::shared_ptr<base::Buffer> FireReader::mapped_buffer() const;
```

`find()` 只做 lookup：未打开或名称不存在均返回 `nullptr`，不返回 `InvalidArgument`。`tensor_count()` 在尚未成功打开时返回 `0`；`mapped_buffer()` 此时返回空 `shared_ptr`。完整 TinyLlama 加载会将缺少 required tensor、额外 tensor 或 dtype/rank/shape profile 不匹配报告为 `ModelParseError`；只有完整 profile 通过后才交付结构化权重，失败时不发布局部组装结果。当前实现已具备该加载逻辑，独立的异常 profile fixture 测试仍待补齐。

`find()` 返回的指针由 Reader 持有，只在该 Reader 存活且未成功重新打开文件期间有效；v0.1 仍按每个文件使用新 Reader 的约定。需要跨越 Reader 生命周期的数据所有权必须通过 `mapped_buffer()` 交给 Tensor。实现不得使用现有会进入 `LOG(FATAL)` 的 `STATUS_CHECK` 宏处理 Reader/ModelLoader 的外部输入错误，而应显式返回相应 `base::Status`。

## 12. 最小验收测试

截至 2026-09-20，本节的 core 合同已自动化：CMake 在构建 `fire_tests` 前生成独立 fixture，FireReader GTest 覆盖有效文件、坏文件矩阵、显式 `uint64` 溢出、Status 分类和 mapping lifetime；CTest 项 `fire_export_tinyllama_contract` 运行 13 个 Writer/exporter unittest，`fire_export_qwen3_contract` 运行 9 个 Qwen3 exporter unittest。这些测试验证 wire 和 profile 合同，不代替本节末尾的真实权重里程碑。

### Wire contract fixture

Core test 使用独立的 stdlib Python `struct.pack` 脚本生成一个精确 260-byte `.fire` fixture，不调用 FireWriter，也不要求安装 Torch：

```text
Header                 32 bytes
2 * TensorInfo        192 bytes
payload                36 bytes
total                  260 bytes
```

fixture 包含：

- `norm.weight`：FP32 `[3]`，包含正数、负数和一个有明确 bit pattern 的 BF16 可表示值。
- `layers.0.attention.wq.weight`：FP32 `[2, 3]`，用于验证 row-major layout 和二维索引。
- `data_offset = 224`，两个 payload 的绝对 offset 分别为 `224` 和 `236`。

C++ GTest 必须验证 Header/TensorInfo 的固定 wire offset、未打开时 lookup、name lookup、missing-name、dtype、shape、byte size、绝对 offset、FP32 bit pattern 和二维布局。还必须构造共享 mmap Tensor，销毁 FireReader 后继续读取，以覆盖 mapping ownership。

错误测试从有效的 260-byte fixture 复制并修改指定 byte，至少覆盖：bad magic/version、错误 directory/data offset、截断 Header/Directory/payload、空名称、未终止名称、首个 NUL 后非零 name byte、重复名称、非零 TensorInfo padding、rank-1 未用 shape 非零、错误 dtype/ndim/shape-byte-size、未对齐或非连续 payload、越界 offset、接近 `UINT64_MAX` 的 checked-arithmetic overflow，以及文件尾不匹配。还需验证 `open()` 的 Status 分类，以及失败后 `find()` 和 `mapped_buffer()` 均为空。

Python writer 测试还应验证：preflight 失败时不创建目标；目标已存在时内容保持不变；注入普通 payload write failure 或 `KeyboardInterrupt` 后，本次 exclusive-create 的 partial file 被删除；成功文件的最终位置和长度严格等于预期值。

### TinyLlama profile test

轻量 Python descriptor test 不生成真实 payload，验证：

- `22 * 9 + 3 = 201` 个 tensor。
- 全部 canonical mapping、FP32 目标 dtype 和预期 shape。
- FP32 payload 合计 `4,400,193,536` bytes，最终文件为 `4,400,212,864` bytes。
- source 缺项、额外项、错误 BF16 dtype、错误 shape 和关键 config 不匹配均会失败。

### Qwen3 profile test

轻量 Python test 不生成真实大文件，验证：

- `28 * 11 + 3 = 311` 与 `36 * 11 + 3 = 399` 两个 descriptor。
- 0.6B 的 Q/K/V/O、Q/K norm 和 FFN canonical name/shape，尤其是 `q_dim != hidden_size`。
- 单文件与 shard index 的完整 tensor set、路径、总字节数和每 shard 实际 key 校验。
- metadata pass 与 payload pass 对每个 shard 只打开一次，且分组后的 absolute offset 连续。
- preflight 失败不创建输出，payload 阶段失败删除本次创建的 partial file。

此外，本地 `models/Qwen3-0.6B` 已确认 311 项、单个 shard、`data_offset = 29,888` 和最终 FP32 文件长度 `3,006,559,424` bytes；该完整文件已通过两 token CPU forward 和 Hugging Face logits 对齐。`models/Qwen3-8B` 完成了 metadata-only preflight，确认 399 项、5 个 shards 和最终预期长度 `32,762,979,776` bytes，尚未写出完整 payload。

### 本地里程碑验收

完整 TinyLlama 导出不进入 core test。本地 release 验收需要执行真实导出，确认 FireReader 解析 201 项、TinyLlama ModelLoader 接受完整 profile，并对 embedding、首尾层、final norm 和 output 的选定元素做 bit-exact FP32 回读。

当前全量 payload 已写出为 `tmp/llama.fire`，其长度为 `4,400,212,864` bytes；FireReader 全量解析、完整 ModelLoader profile 校验以及真实 CPU/GPU forward 均已执行。选定 tensor 与 source checkpoint 的独立 bit-exact 抽样回读仍缺少完整记录；已有 Loader、Norm 与 forward 集成测试不能替代这一项来源对照。
