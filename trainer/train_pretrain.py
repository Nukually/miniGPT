import os
import sys
import warnings
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_REPO_ROOT)

from dataset.lm_dataset import PretrainDataset
from model.model_minigpt import MiniGPTConfig
from trainer.trainer_utils import (
    is_main_process,
    Logger,
    init_distributed_mode,
    setup_seed,
    lm_checkpoint,
    init_model,
    SkipBatchSampler,
    train_epoch
)

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    class TrainingArgs:
        save_dir = "../out"
        save_weight = "pretrain"
        epochs = 1
        batch_size = 32
        learning_rate = 5e-4
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16"
        num_workers = 8
        accumulation_steps = 8
        grad_clip = 1.0
        log_interval = 100
        save_interval = 1000
        hidden_size = 512
        num_hidden_layers = 8
        max_seq_len = 340
        use_moe = 0
        data_path = "../dataset/pretrain_hq.jsonl"
        from_weight = "none"
        from_resume = 0
        use_compile = 0
        use_wandb = False
        wandb_project = "MiniGPT-Pretrain"

    args = TrainingArgs()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        # DDP中每个进程绑定一张GPU
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniGPTConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if args.from_resume == 1
        else None
    )
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniGPT-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 只有float16需要GradScaler；bf16下无需缩放
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型、优化器与混合精度状态
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, model, optimizer, scaler, args, lm_config, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), model, optimizer, scaler, args, lm_config, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
