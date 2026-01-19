import argparse
import os
import time
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from model.model_minigpt import MiniGPTConfig, MiniGPTForCausalLM
from trainer.trainer_utils import get_model_params, setup_seed
warnings.filterwarnings("ignore")
DEFAULT_SEED = 2026

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if "model" in args.load_from:
        model = MiniGPTForCausalLM(
            MiniGPTConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )
        moe_suffix = '_moe' if args.use_moe else ''
        weight_name = f'{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        ckp = os.path.join(args.save_dir, weight_name)
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    conversation = []
    while True:
        try:
            prompt = input('💬: ')
        except EOFError:
            break
        if not prompt:
            break

        setup_seed(DEFAULT_SEED)
        # 只保留最近历史，控制上下文长度。
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = True  # 仅Reason模型使用
        if args.weight != 'pretrain':
            inputs = tokenizer.apply_chat_template(**templates)
        else:
            # 预训练权重使用原始文本，不走对话模板。
            inputs = tokenizer.bos_token + prompt
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=1.0,
            )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n')
        else:
            print('\n\n')


if __name__ == "__main__":
    main()
