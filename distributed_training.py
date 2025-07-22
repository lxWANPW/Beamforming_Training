#!/usr/bin/env python3
"""
分布式训练脚本 - 使用DistributedDataParallel进行更高效的多GPU训练
使用方法：
python -m torch.distributed.launch --nproc_per_node=2 distributed_training.py
或者
torchrun --nproc_per_node=2 distributed_training.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from beamforming_model_fixed_v2 import *
import argparse

def setup_distributed():
    """设置分布式训练"""
    # 获取环境变量
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置CUDA设备
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """清理分布式训练"""
    dist.destroy_process_group()

def distributed_train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-4, local_rank=0, rank=0):
    """分布式训练函数"""
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.95)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=lr/100
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练参数
    accumulation_steps = 2
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    
    if rank == 0:
        print(f"🚀 开始分布式训练 (Rank {rank}):")
        print(f"   • 初始学习率: {lr}")
        print(f"   • 设备: {device}")
        print(f"   • 世界大小: {dist.get_world_size()}")
    
    for epoch in range(num_epochs):
        # 设置epoch用于分布式采样器
        train_loader.sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_theta_loss = 0
        train_phi_loss = 0
        num_batches = 0
        
        if rank == 0:
            train_progress = tqdm(train_loader, desc=f"🎯 训练 Epoch {epoch+1}/{num_epochs}", leave=False)
        else:
            train_progress = train_loader
        
        optimizer.zero_grad()
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = ultra_loss_function(
                        predictions, (theta_label, phi_label), loss_type='ultra_focal'
                    )
                    loss = loss / accumulation_steps
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积和优化
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss_epoch += loss.item() * accumulation_steps
                train_theta_loss += theta_loss.item()
                train_phi_loss += phi_loss.item()
                num_batches += 1
                
                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    train_progress.set_postfix({
                        'Loss': f'{loss.item() * accumulation_steps:.4f}',
                        'θ': f'{theta_loss.item():.4f}',
                        'φ': f'{phi_loss.item():.4f}',
                        'LR': f'{current_lr:.2e}'
                    })
                
            except Exception as e:
                if rank == 0:
                    print(f"❌ 训练批次 {batch_idx} 出错: {e}")
                continue
        
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_theta_loss = 0
        val_phi_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            if rank == 0:
                val_progress = tqdm(val_loader, desc="🔍 验证", leave=False)
            else:
                val_progress = val_loader
                
            for H_input, theta_label, phi_label, H_complex in val_progress:
                try:
                    H_input = H_input.to(device, non_blocking=True)
                    theta_label = theta_label.to(device, non_blocking=True)
                    phi_label = phi_label.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        predictions = model(H_input)
                        loss, theta_loss, phi_loss = ultra_loss_function(
                            predictions, (theta_label, phi_label), loss_type='ultra_focal'
                        )
                    
                    val_loss_epoch += loss.item()
                    val_theta_loss += theta_loss.item()
                    val_phi_loss += phi_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        # 同步所有进程的验证损失
        if val_batches > 0:
            val_loss_tensor = torch.tensor([val_loss_epoch / val_batches], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss_avg = val_loss_tensor.item()
        else:
            val_loss_avg = float('inf')
        
        # 训练损失
        if num_batches > 0:
            train_loss_tensor = torch.tensor([train_loss_epoch / num_batches], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss_avg = train_loss_tensor.item()
        else:
            train_loss_avg = float('inf')
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # 只在主进程打印和保存
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🔥 Epoch {epoch+1:3d}: "
                  f"Train={train_loss_avg:.6f} | "
                  f"Val={val_loss_avg:.6f} | "
                  f"LR={current_lr:.2e}")
            
            # 保存最佳模型
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                torch.save(model.module.state_dict(), 'best_distributed_model.pth')
                print(f"   ✅ 保存最佳模型 (Val Loss: {val_loss_avg:.6f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"🛑 早停触发！在epoch {epoch+1}停止训练")
                break
        
        # 同步所有进程
        dist.barrier()
    
    return train_losses, val_losses

def main_distributed():
    """分布式训练主函数"""
    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed()
    
    try:
        # 只在主进程打印初始信息
        if rank == 0:
            print(f"🖥️  分布式训练初始化:")
            print(f"   • 总进程数: {world_size}")
            print(f"   • 当前rank: {rank}")
            print(f"   • 本地rank: {local_rank}")
        
        # 数据路径
        channel_folder = "samples_data/channel_data_opt_20250702_143327"
        angle_folder = "samples_data/angle_data_opt_20250702_143327"
        
        # 创建数据集
        dataset = BeamformingDataset(channel_folder, angle_folder)
        
        # 数据集划分
        total_samples = len(dataset)
        train_size = 800
        indices = list(range(total_samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        val_size = int(train_size * 0.25)
        train_final_indices = train_indices[val_size:]
        val_indices = train_indices[:val_size]
        
        # 创建数据子集
        train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # 分布式采样器
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # 数据加载器
        batch_size = 4  # 每个进程的批次大小
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型
        if rank == 0:
            print("🏗️  创建增强模型...")
        model = create_enhanced_model()
        
        # 分布式训练
        if rank == 0:
            print("🚀 开始分布式训练...")
        
        train_losses, val_losses = distributed_train_model(
            model, train_loader, val_loader,
            num_epochs=200,
            lr=1e-4,
            local_rank=local_rank,
            rank=rank
        )
        
        if rank == 0:
            print("🏁 分布式训练完成！")
    
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分布式波束成形训练')
    parser.add_argument('--local_rank', type=int, default=0, help='本地GPU rank')
    args = parser.parse_args()
    
    main_distributed()
