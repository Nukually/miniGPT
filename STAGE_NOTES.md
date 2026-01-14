# 开发阶段记录

本文档用于记录每个阶段的代码实现方式，方便后续维护与复盘。所有命令默认在仓库根目录执行，并使用 `minimind` conda 环境。

## 阶段一：Tokenizer 训练与验证

**目标**：从 `dataset/pretrain_hq.jsonl` 训练 BPE 分词器，并完成最小化的编码/解码验证。

**实现文件**
- `trainer/train_tokenizer.py`：训练并保存 tokenizer。
- `tests/test_phase1.py`：加载 `model/` 下的 tokenizer 做 round-trip 测试。

**核心实现**
- 使用 `tokenizers` 的 `BPE` 模型，搭配 `ByteLevel` 预分词器与解码器。
- 训练时读取 JSONL 中的 `text` 字段，可通过 `--limit` 控制样本量。
- 默认加入特殊 token：`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`。
- 使用 `PreTrainedTokenizerFast` 保存到 `model/`。

**命令示例**
```bash
conda run -n minimind python trainer/train_tokenizer.py --overwrite --limit 20000
conda run -n minimind python tests/test_phase1.py
```

**验证标准**
- `tests/test_phase1.py` 能输出原文、token ids、解码文本。
- 解码文本（去除空格）与原文一致，即通过。

## 阶段二：模型结构 (MiniGPT)

**目标**：实现最小可运行的 Transformer 语言模型，并完成前向传播验证。

**实现文件**
- `model/model_minigpt.py`：模型结构实现（RMSNorm、RoPE、Attention、FFN、MiniGPT）。
- `model/__init__.py`：导出模型与配置，便于模块化引用。
- `tests/test_phase2.py`：小配置前向验证。

**核心实现**
- 使用 RMSNorm 作为归一化方式。
- 使用 RoPE (Rotary Positional Embedding) 为注意力注入位置信息。
- Attention 采用多头自注意力 + causal mask，确保仅关注历史 token。
- FFN 采用 SwiGLU 结构（`SiLU(gate) * up`），并可切换为 MoE。
- MoE 通过门控网络对多个专家前馈层进行加权融合，支持 `moe_num_experts` 与 `moe_top_k`。
- 输出层与词嵌入权重共享，节省参数。

**命令示例**
```bash
conda run -n minimind python tests/test_phase2.py
```

**验证标准**
- 输出形状为 `[batch, seq_len, vocab_size]`。
- 不报错且打印 “Phase 2 check passed”。
