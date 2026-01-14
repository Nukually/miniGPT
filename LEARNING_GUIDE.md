# MiniGPT 从零搭建完全指南：循序渐进版

本指南旨在帮助你以**学习**为目的，从零开始搭建属于你自己的大语言模型 —— **MiniGPT**。
（注：本项目参考了 MiniMind 的设计思路，但我们将从头构建一个全新的 MiniGPT 模型）

为了让你学得更扎实，我们将流程拆分为 10 个里程碑。
**每个里程碑完成后，都配有「验证环节」和「当前项目结构图」，确保你每一步都走得稳当。**

---

## 🚀 阶段一：环境搭建与分词器 (Tokenizer)

**目标**：配置好 Python 环境，并让机器能够把文字转换成数字（Token）。

### 1.1 准备工作
你需要安装 `torch`, `transformers`, `datasets` 等基础库。

### 1.2 核心任务
1.  准备一份纯文本数据（例如 `dataset/pretrain_hq.jsonl` 中的文本）。
2.  使用 `tokenizers` 库训练一个 BPE 分词器。
3.  或者直接使用项目提供的 `tokenizer.json`。

### 🧪 验证环节
在项目根目录下创建一个测试脚本 `test_phase1.py`：

```python
from transformers import PreTrainedTokenizerFast

# 加载分词器 (假设你放在了 ./model/ 目录下)
tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")

text = "你好，MiniGPT！"
input_ids = tokenizer.encode(text)
decoded_text = tokenizer.decode(input_ids)

print(f"原文: {text}")
print(f"Token IDs: {input_ids}")
print(f"还原: {decoded_text}")

# 验证一致性
assert text == decoded_text.replace(" ", "") # 注意：某些tokenizer会加空格，视情况调整
print("✅ 阶段一验证成功！分词器工作正常。")
```

### 📂 当前项目结构
```text
minigpt/
├── dataset/
│   └── pretrain_hq.jsonl  # 原始数据
├── model/
│   ├── tokenizer.json     # 核心词表文件
│   └── tokenizer_config.json
└── test_phase1.py         # 刚才的测试脚本
```

---

## 🏗️ 阶段二：模型构建 (Model Architecture)

**目标**：手写一个 Transformer 模型（MiniGPT），而不是直接 import。

### 2.1 核心任务
创建 `model/model_minigpt.py`。你需要实现：
1.  `RMSNorm`: 归一化层。
2.  `RoPE`: 旋转位置编码（这是 LLM 支持长文本的关键）。
3.  `Attention`: 自注意力机制。
4.  `FeedForward`: 前馈网络 (SwiGLU)。
5.  `MiniGPT`: 组合以上模块。

### 🧪 验证环节
创建 `test_phase2.py`，检查模型能不能跑通一次“前向传播”：

```python
import torch
from model.model_minigpt import MiniGPT, MiniGPTConfig

# 1. 初始化配置 (使用极小配置以快速测试)
config = MiniGPTConfig(
    vocab_size=6400,
    hidden_size=256,   # 小一点方便CPU测
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=512
)

# 2. 实例化模型
model = MiniGPT(config)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

# 3. 构造虚拟输入 (Batch=2, SeqLen=10)
dummy_input = torch.randint(0, 6400, (2, 10))

# 4. 前向传播
output = model(dummy_input)

# 5. 检查输出形状
# 期望输出: [Batch, SeqLen, VocabSize]
expected_shape = (2, 10, 6400)
assert output.logits.shape == expected_shape
print(f"输出形状: {output.logits.shape}")
print("✅ 阶段二验证成功！模型结构搭建完毕，输入输出对齐。")
```

### 📂 当前项目结构
```text
minigpt/
├── dataset/ ...
├── model/
│   ├── __init__.py
│   ├── model_minigpt.py  # <--- 新增核心代码
│   └── tokenizer...
├── test_phase1.py
└── test_phase2.py         # <--- 新增测试
```

---

## 📚 阶段三：数据管道 (Dataset Pipeline)

**目标**：把原始文本处理成模型能吃的 `Tensor`，特别是要搞懂 **Mask**。

### 3.1 核心任务
创建 `dataset/lm_dataset.py`。
1.  **PretrainDataset**: 简单的滑窗截断。输入 `x` 是 `[0:-1]`, 标签 `y` 是 `[1:]`。
2.  **SFTDataset**: **(重难点)** 处理对话格式。
    *   构造 Input: `<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n我是MiniGPT<|im_end|>`
    *   构造 Mask: 只有“我是MiniGPT”这部分的 loss 应该被计算，user 的提问部分 loss mask 设为 0。

### 🧪 验证环节
创建 `test_phase3.py`，肉眼检查 Mask 对不对：

```python
from transformers import PreTrainedTokenizerFast
from dataset.lm_dataset import SFTDataset

tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")

# 模拟一个 SFT 数据文件
import json
with open("test_sft.jsonl", "w", encoding="utf-8") as f:
    data = {
        "conversations": [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"}
        ]
    }
    f.write(json.dumps(data, ensure_ascii=False))

# 加载数据集
ds = SFTDataset("test_sft.jsonl", tokenizer, max_length=64)
x, y, mask = ds[0]

print("Input:", tokenizer.decode(x))
print("Mask :", mask.tolist())

# 简单验证: user部分(A)的mask应该是0, assistant部分(B)的mask应该是1
# 注意: 不同tokenizer处理特殊字符方式不同，建议打印出来肉眼确认 '1' 覆盖了回答部分
print("✅ 阶段三验证完成！请人工确认 Mask 是否覆盖了回答部分。")
```

### 📂 当前项目结构
```text
minigpt/
├── dataset/
│   ├── pretrain_hq.jsonl
│   └── lm_dataset.py      # <--- 新增数据处理逻辑
├── model/ ...
├── test_sft.jsonl         # 临时测试文件
└── test_phase3.py         # <--- 新增测试
```

---

## 🏋️ 阶段四：预训练循环 (Pretraining Loop)

**目标**：写出训练循环，让 Loss 动起来。

### 4.1 核心任务
创建 `trainer/train_pretrain.py`。
1.  加载 Model 和 Dataset。
2.  初始化 Optimizer (AdamW)。
3.  编写 Loop: `Forward` -> `Loss` -> `Backward` -> `Step`。
4.  保存模型权重 (`.pth` 或 `.safetensors`)。

### 🧪 验证环节
直接运行训练脚本，但参数设得很小，只跑几步：

```bash
# 命令行测试
python trainer/train_pretrain.py --epochs 1 --batch_size 2 --save_dir ./out_test
```

**检查点**：
1.  终端是否打印出 Loss (例如 `loss: 8.5432`)？
2.  Loss 是否不是 `NaN`？
3.  `./out_test` 目录下是否生成了 `.pth` 文件？

### 📂 当前项目结构
```text
minigpt/
├── dataset/ ...
├── model/ ...
├── trainer/
│   └── train_pretrain.py  # <--- 新增训练脚本
├── out_test/              # <--- 生成的权重目录
│   └── pretrain_xxx.pth
└── ...
```

---

## 🗣️ 阶段五：监督微调 (SFT)

**目标**：让模型学会对话格式，不再胡言乱语。

### 5.1 核心任务
创建 `trainer/train_full_sft.py`。
*   逻辑与预训练几乎一样，但加载的是 `SFTDataset`。
*   需要加载**阶段四**训练好的预训练权重作为起点 (Init from Pretrain)。

### 🧪 验证环节
同样运行一个小测试：
```bash
python trainer/train_full_sft.py --epochs 1 --batch_size 2 --save_dir ./out_sft_test
```
确认 Loss 下降，且保存了新的权重。

### 📂 当前项目结构
```text
minigpt/
├── dataset/ ...
├── model/ ...
├── trainer/
│   ├── train_pretrain.py
│   └── train_full_sft.py  # <--- 新增 SFT 脚本
├── out_sft_test/          # <--- SFT 权重
└── ...
```

---

## 🤖 阶段六：推理与对话 (Inference)

**目标**：见证奇迹的时刻，和你的模型聊天。

### 6.1 核心任务
创建 `scripts/web_demo.py` 或简单的 `chat.py`。
1.  加载 SFT 后的权重。
2.  实现 `generate` 函数（如果是手写的）或调用 `model.generate`。
3.  处理 `Input` -> `Tokenizer` -> `Model` -> `Tokenizer` -> `Output` 的流向。

### 🧪 验证环节
创建 `test_chat.py`：

```python
import torch
from transformers import PreTrainedTokenizerFast
from model.model_minigpt import MiniGPT, MiniGPTConfig

# 1. 加载配置和模型
tokenizer = PreTrainedTokenizerFast.from_pretrained("./model")
model = MiniGPT(MiniGPTConfig(...)) # 填入你的配置
# 加载你训练好的 SFT 权重
state_dict = torch.load("./out_sft_test/xxx.pth", map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. 对话
prompt = "你好"
messages = [{"role": "user", "content": prompt}]
input_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(input_str, return_tensors='pt').input_ids

with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=50)
    
response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
print(f"User: {prompt}")
print(f"MiniGPT: {response}")

print("✅ 阶段六验证完成！如果回复通顺，恭喜你复现成功！")
```

### 📂 当前项目结构
```text
minigpt/
├── dataset/         # 数据处理
├── model/           # 模型定义 & Tokenizer
├── trainer/         # 训练脚本 (Pretrain, SFT)
├── scripts/         # 推理 & Demo
├── out/             # 存放训练好的权重
└── tests/           # (推荐) 存放所有的测试脚本
```

---

## 🏎️ 阶段七：LoRA 微调 (Low-Rank Adaptation)

**目标**：以极小的显存代价（几MB参数）微调大模型。

### 7.1 核心任务
1.  **Model (LoRA)**: 创建 `model/model_lora.py`。
    *   定义 `LoRA` 类：包含两个低秩矩阵 A 和 B。
    *   定义 `apply_lora` 函数：遍历模型所有 `Linear` 层，将其替换为带 LoRA 的版本。
2.  **Trainer (LoRA)**: 创建 `trainer/train_lora.py`。
    *   加载预训练/SFT权重。
    *   调用 `apply_lora` 注入参数。
    *   **关键点**：仅将 LoRA 参数设为 `requires_grad=True`，冻结其他参数。

### 🧪 验证环节
创建 `test_phase7.py`：

```python
import torch
from model.model_minigpt import MiniGPT, MiniGPTConfig
from model.model_lora import apply_lora

model = MiniGPT(MiniGPTConfig(hidden_size=256, num_hidden_layers=2))
print(f"原始参数量: {sum(p.numel() for p in model.parameters())}")

# 应用 LoRA
apply_lora(model, rank=8)

# 检查可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA后可训练参数量: {trainable_params}")

# 简单前向传播
x = torch.randint(0, 100, (1, 10))
y = model(x)
print("✅ 阶段七验证成功！LoRA 已注入且前向传播正常。")
```

---

## 🧠 阶段八：推理模型 (Reasoning / Chain of Thought)

**目标**：让模型学会“慢思考”，输出 `<think>` 标签。

### 8.1 核心任务
创建 `trainer/train_reason.py`。
1.  **Dataset**: 准备包含思考过程的数据（如 `<think>...过程...</think><answer>...结果...</answer>`）。
2.  **Loss Weighting**: 这是一个关键技巧。
    *   为了强迫模型学会使用标签，在计算 Loss 时，给 `<think>`, `</think>`, `<answer>`, `</answer>` 这些特殊 token **加权**（例如 10倍权重）。
    *   这能防止模型“偷懒”跳过思考过程。

### 🧪 验证环节
查看 `train_reason.py` 中的 `loss_mask` 处理逻辑。
可以手动构造一个 Batch，检查 loss_mask 中对应 `<think>` 的位置是否真的是 10。

---

## 👮 阶段九：RLHF (PPO) —— 人类偏好对齐

**目标**：使用强化学习让模型更符合人类价值观。

### 9.1 核心任务
创建 `trainer/train_ppo.py`。
1.  **Critic Model**: 基于 `MiniGPT` 增加一个 `value_head`，输出标量价值。
2.  **Reward Function**:
    *   **Format Reward**: 检查输出是否符合格式（如包含 `<think>` 标签），符合给分，不符合扣分。
    *   **Model Reward**: 使用另一个训练好的 Reward Model 打分。
3.  **PPO Step**:
    *   计算 `Advantage` (GAE)。
    *   计算 `Policy Loss` (Clipping)。
    *   计算 `Value Loss`。

## 🏆 阶段十：GRPO (Group Relative Policy Optimization) —— DeepSeek-R1 同款算法

**目标**：抛弃 Critic 模型，直接使用组内相对奖励来优化 Policy，大幅降低显存占用。

### 10.1 核心任务
创建 `trainer/train_grpo.py`。这是 DeepSeek-R1 提出的核心算法。
1.  **Group Sampling**:
    *   对于每一个 Prompt，让 Policy Model 采样生成 $G$ 个不同的 Responses (e.g., $G=4$)。
2.  **Reward Calculation**:
    *   对这 $G$ 个回复分别计算 Reward (规则分 + 模型分)。
    *   计算组内平均分 $\mu$ 和标准差 $\sigma$。
    *   计算优势函数 (Advantage): $A_i = \frac{R_i - \mu}{\sigma}$。
3.  **Optimization**:
    *   最大化 $E[\frac{\pi(y|x)}{\pi_{old}(y|x)} \cdot A_i]$。
    *   同时添加 KL 散度约束，防止偏离 Reference Model 太远。

### 10.2 GRPO vs PPO
*   **PPO**: 需要 4 个模型 (Actor, Critic, Ref, Reward)。显存占用巨大。
*   **GRPO**: 只需要 2 个模型 (Actor, Ref)。Reward 可以是简单的规则函数（如数学题判卷）。显存极度节省，且效果往往更好。

### 🧪 验证环节
运行 `trainer/train_grpo.py`。
观察日志中生成的 Responses，你会发现随着训练进行，模型开始更倾向于生成带有 `<think>` 标签且得分更高的回答。

### 📂 最终完全体项目结构
```text
minigpt/
├── dataset/         # 数据处理 (Pretrain, SFT, DPO, RLHF)
├── model/           # 模型定义 (MiniGPT, LoRA, Critic)
├── trainer/         # 训练脚本 (Pretrain, SFT, LoRA, Reason, PPO, GRPO)
├── scripts/         # 推理 & Demo
├── out/             # 存放训练好的权重
└── tests/           # 测试脚本
```

---

按照这个结构一步步来，每一步都运行测试代码验证，你将不会迷失在复杂的代码中。祝你 coding 愉快！
