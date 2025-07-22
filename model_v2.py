import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import h5py
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from tqdm import tqdm
import warnings
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import subprocess
import os
from torch.utils.checkpoint import checkpoint
import time

class BeamformingDataset(Dataset):
    """优化版天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None, use_augmentation=True):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # 获取信道数据文件列表并排序
        self.channel_files = sorted(glob.glob(os.path.join(channel_folder, '*.mat')))
        print(f"找到信道数据文件: {len(self.channel_files)} 个")
        
        # 加载角度数据（所有样本的角度信息在一个文件中）
        angle_files = glob.glob(os.path.join(angle_folder, '*.mat'))
        if len(angle_files) > 0:
            try:
                # 尝试使用 h5py 读取
                with h5py.File(angle_files[0], 'r') as f:
                    all_theta = np.array(f['all_theta'])  # (50, 1000) or (1000, 50)
                    all_phi = np.array(f['all_phi'])      # (50, 1000) or (1000, 50)
                    
                    # 检查维度并调整
                    if all_theta.shape[0] == 50 and all_theta.shape[1] == 1000:
                        self.theta_array = all_theta.T  # (1000, 50)
                        self.phi_array = all_phi.T      # (1000, 50)
                    else:
                        self.theta_array = all_theta    # 假设已经是(1000, 50)
                        self.phi_array = all_phi
                    
                    self.num_samples = self.theta_array.shape[0]
                    self.num_time_samples = self.theta_array.shape[1]
                    
                    print(f"角度数据形状: theta={self.theta_array.shape}, phi={self.phi_array.shape}")
                    
            except Exception as e:
                print(f"h5py 读取失败: {e}")
                # 使用 scipy.io 读取
                try:
                    angle_data = sio.loadmat(angle_files[0])
                    # 检查可能的键名
                    possible_keys = ['all_theta', 'theta_array', 'theta']
                    theta_key = None
                    phi_key = None
                    
                    for key in angle_data.keys():
                        if 'theta' in key.lower():
                            theta_key = key
                        if 'phi' in key.lower():
                            phi_key = key
                    
                    if theta_key and phi_key:
                        all_theta = angle_data[theta_key]
                        all_phi = angle_data[phi_key]
                        
                        if all_theta.shape[0] == 50 and all_theta.shape[1] == 1000:
                            self.theta_array = all_theta.T
                            self.phi_array = all_phi.T
                        else:
                            self.theta_array = all_theta
                            self.phi_array = all_phi
                            
                        self.num_samples = self.theta_array.shape[0]
                        self.num_time_samples = self.theta_array.shape[1]
                    else:
                        raise ValueError("未找到theta和phi数据")
                        
                except Exception as e2:
                    print(f"scipy.io 读取也失败: {e2}")
                    print(f"角度文件内容: {list(angle_data.keys()) if 'angle_data' in locals() else 'Unknown'}")
                    raise e2
        
        # 确保数据一致性，使用较小的数量
        self.actual_num_samples = min(len(self.channel_files), self.num_samples)
        print(f"实际使用的样本数量: {self.actual_num_samples}")
        
    def __len__(self):
        return self.actual_num_samples
    
    def _add_data_augmentation(self, H_real, H_imag, theta_rad, phi_rad):
        """轻量级数据增强策略"""
        if not self.use_augmentation:
            return H_real, H_imag, theta_rad, phi_rad
            
        # 1. 轻微噪声增强
        noise_std = 0.005  # 减小噪声强度
        H_real = H_real + np.random.normal(0, noise_std, H_real.shape)
        H_imag = H_imag + np.random.normal(0, noise_std, H_imag.shape)
        
        # 2. 角度小幅抖动
        angle_noise_std = 0.01  # 约0.6度的标准差
        theta_rad = theta_rad + np.random.normal(0, angle_noise_std, theta_rad.shape)
        phi_rad = phi_rad + np.random.normal(0, angle_noise_std, phi_rad.shape)
        
        return H_real, H_imag, theta_rad, phi_rad
    
    def __getitem__(self, idx):
        if idx >= self.actual_num_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.actual_num_samples}")
            
        # 加载信道数据
        try:
            # 尝试使用 h5py 读取
            with h5py.File(self.channel_files[idx], 'r') as f:
                H_real = np.array(f['H_real'])  # (50, 256, 256)
                H_imag = np.array(f['H_imag'])  # (50, 256, 256)
                
                # 转置以匹配预期格式 [256, 256, 50]
                H_real = H_real.transpose(1, 2, 0)  # (256, 256, 50)
                H_imag = H_imag.transpose(1, 2, 0)  # (256, 256, 50)
                
        except Exception as e:
            # 使用 scipy.io 读取
            try:
                channel_data = sio.loadmat(self.channel_files[idx])
                H_real = channel_data['H_real']  # [256, 256, 50]
                H_imag = channel_data['H_imag']
            except Exception as e2:
                print(f"无法读取信道数据文件 {self.channel_files[idx]}: {e2}")
                raise e2
        
        # 获取角度数据
        theta_rad = self.theta_array[idx] * np.pi / 180  # 转换为弧度
        phi_rad = self.phi_array[idx] * np.pi / 180
        
        # 轻量级数据增强
        H_real, H_imag, theta_rad, phi_rad = self._add_data_augmentation(
            H_real, H_imag, theta_rad, phi_rad
        )
        
        # 组合实部和虚部
        H_complex = H_real + 1j * H_imag
        
        # 更高效的数据归一化
        H_real_norm = H_real / (np.std(H_real) + 1e-8)
        H_imag_norm = H_imag / (np.std(H_imag) + 1e-8)
        
        # 裁剪极值
        H_real_norm = np.clip(H_real_norm, -6, 6)
        H_imag_norm = np.clip(H_imag_norm, -6, 6)
        
        # 转换为PyTorch tensor（使用float16减少内存）
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)  # [2, 256, 256, 50]
        H_input = torch.FloatTensor(H_input)
        
        # 角度标签归一化
        theta_norm = np.clip(theta_rad / (np.pi / 2), -1, 1)
        phi_norm = np.clip(phi_rad / np.pi, -1, 1)
        
        theta_label = torch.FloatTensor(theta_norm)  # [50]
        phi_label = torch.FloatTensor(phi_norm)      # [50]
        
        return H_input, theta_label, phi_label, H_complex

class MemoryEfficientPositionalEncoding(nn.Module):
    """内存高效的位置编码"""
    def __init__(self, d_model, max_len=100):
        super(MemoryEfficientPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class EfficientFeatureExtractor(nn.Module):
    """内存高效的特征提取器"""
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(EfficientFeatureExtractor, self).__init__()
        
        # 分阶段降维，大幅减少中间维度
        intermediate_dims = [input_dim, input_dim//8, input_dim//32, d_model*2, d_model]
        
        self.feature_layers = nn.ModuleList()
        for i in range(len(intermediate_dims)-1):
            self.feature_layers.append(
                nn.Sequential(
                    nn.Conv1d(intermediate_dims[i], intermediate_dims[i+1], kernel_size=1),
                    nn.BatchNorm1d(intermediate_dims[i+1]),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
    def forward(self, x):
        for layer in self.feature_layers:
            x = layer(x)
        return x

class OptimizedTransformerBeamformingNet(nn.Module):
    """内存优化的Transformer波束成形网络"""
    def __init__(self, input_dim=256*256*2, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(OptimizedTransformerBeamformingNet, self).__init__()
        
        self.d_model = d_model
        
        # 大幅减小的特征提取器
        self.feature_extractor = EfficientFeatureExtractor(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # 位置编码
        self.pos_encoder = MemoryEfficientPositionalEncoding(d_model, max_len=100)
        
        # 轻量级Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 减小FFN大小
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 简化的注意力聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 轻量级预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 2)  # 直接输出theta和phi
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
                    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, features]
        
        # 特征提取
        x = x.permute(0, 2, 1)  # [batch, features, time_steps]
        x = self.feature_extractor(x)  # [batch, d_model, time_steps]
        x = x.permute(0, 2, 1)  # [batch, time_steps, d_model]
        
        # 位置编码
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码（移除梯度检查点避免DDP问题）
        encoded = self.transformer_encoder(x)
        
        # 全局池化
        encoded = encoded.permute(0, 2, 1)  # [batch, d_model, time_steps]
        pooled = self.global_pool(encoded).squeeze(-1)  # [batch, d_model]
        
        # 预测
        angles = self.predictor(pooled)  # [batch, 2]
        
        return angles

class FocalHuberLoss(nn.Module):
    """简化的损失函数"""
    def __init__(self, alpha=2.0, delta=0.5):
        super(FocalHuberLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        
    def forward(self, predictions, targets):
        target_theta, target_phi = targets
        
        # 使用最后一个时间步的目标
        target_theta_last = target_theta[:, -1]  # [batch]
        target_phi_last = target_phi[:, -1]      # [batch]
        
        # Huber损失
        theta_loss = F.huber_loss(predictions[:, 0], target_theta_last, delta=self.delta)
        phi_loss = F.huber_loss(predictions[:, 1], target_phi_last, delta=self.delta)
        
        total_loss = self.alpha * (theta_loss + phi_loss)
        
        return total_loss, theta_loss, phi_loss

def setup_distributed_training():
    """设置分布式训练环境"""
    # 检查是否使用multiprocessing.spawn方式
    if os.environ.get('USE_MP_SPAWN') == '1' and torch.cuda.device_count() > 1:
        print("使用 multiprocessing.spawn 启动分布式训练")
        
        # 设置环境变量
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        # 启动多进程
        if not os.environ.get('SPAWN_STARTED'):
            os.environ['SPAWN_STARTED'] = '1'
            mp.spawn(
                distributed_main_spawn,
                args=(),
                nprocs=torch.cuda.device_count(),
                join=True
            )
            return None, None, None, True
    
    # 如果是单机多卡，自动设置环境变量
    if 'WORLD_SIZE' not in os.environ and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个GPU，自动启用分布式训练")
        
        # 自动设置环境变量
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        # 启动其他进程
        if os.environ.get('AUTO_DISTRIBUTED') != 'True':
            os.environ['AUTO_DISTRIBUTED'] = 'True'
            
            # 使用torch.multiprocessing启动多进程
            mp.spawn(
                distributed_main,
                args=(),
                nprocs=torch.cuda.device_count(),
                join=True
            )
            return None, None, None, True  # 标记为已处理
    
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"初始化分布式训练: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        
        # 初始化进程组
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank,
                    timeout=torch.distributed.constants.default_pg_timeout
                )
                print(f"Rank {rank}: 分布式进程组初始化完成")
            except Exception as e:
                print(f"分布式初始化失败: {e}")
                return 1, 0, 0, False
        
        return world_size, rank, local_rank, False
    else:
        return 1, 0, 0, False

def distributed_main_spawn(rank):
    """使用spawn方式的分布式训练主函数"""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # 防止递归调用
    if 'SPAWN_STARTED' in os.environ:
        del os.environ['USE_MP_SPAWN']
    
    main()

def distributed_main(rank):
    """分布式训练主函数"""
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    
    main()

def train_optimized_model(model, train_loader, val_loader, num_epochs=150, lr=1e-4, 
                         device='cuda', use_distributed=False, world_size=1, rank=0, local_rank=0):
    """优化的模型训练函数"""
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.to(device)
    
    # 包装为分布式模型
    if use_distributed and dist.is_initialized() and world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False,  # 先设为False，强制所有参数参与计算
            broadcast_buffers=False,      # 减少通信开销
            gradient_as_bucket_view=True, # 内存优化
            static_graph=True             # 使用静态图优化
        )
        print(f"Rank {rank}: 模型已包装为DDP（静态图模式）")
    
    # 使用混合精度训练
    scaler = GradScaler()
    
    # 调整学习率
    effective_lr = lr * world_size if use_distributed and world_size > 1 else lr
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=effective_lr, 
        weight_decay=5e-5,  # 减小权重衰减
        eps=1e-8
    )
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=effective_lr * 0.01
    )
    
    # 简化的损失函数
    criterion = FocalHuberLoss(alpha=2.0, delta=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    min_improvement = 1e-6  # 最小改进阈值
    
    # 分布式训练的梯度累积设置
    accumulation_steps = 4 if not use_distributed else 2  # 分布式时减少累积步数
    
    is_main_process = rank == 0
    if is_main_process:
        print(f"开始优化模型训练，有效学习率: {effective_lr}")
        if use_distributed and world_size > 1:
            print(f"使用 {world_size} 个GPU进行分布式训练")
            print(f"梯度累积步数: {accumulation_steps}")
    
    for epoch in range(num_epochs):
        # 设置分布式sampler的epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}", 
                             leave=False, disable=not is_main_process)
        
        # 初始化梯度
        optimizer.zero_grad()
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 检查输入数据有效性
                if torch.isnan(H_input).any() or torch.isinf(H_input).any():
                    if is_main_process:
                        print(f"跳过无效数据批次 {batch_idx}")
                    continue
                
                # 计算是否是累积的最后一步
                is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
                is_last_batch = batch_idx == len(train_loader) - 1
                should_step = is_accumulation_step or is_last_batch
                
                # 前向传播
                with torch.amp.autocast('cuda'):
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = criterion(predictions, (theta_label, phi_label))
                    
                    # 对分布式训练进行loss缩放
                    if use_distributed and world_size > 1:
                        loss = loss / world_size
                    loss = loss / accumulation_steps
                
                # 检查损失有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process:
                        print(f"跳过无效损失批次 {batch_idx}")
                    continue
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 在累积完成时执行优化步骤
                if should_step:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss_epoch += loss.item() * accumulation_steps
                num_batches += 1
                
                # 更新进度条
                train_progress.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.5f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'Step': f'{batch_idx % accumulation_steps + 1}/{accumulation_steps}'
                })
                
                # 定期清理缓存
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if is_main_process:
                    print(f"训练批次 {batch_idx} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 发生错误时重置梯度
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        # epoch结束时确保梯度已清理
        optimizer.zero_grad()
        
        # 同步分布式训练的损失统计
        if use_distributed and dist.is_initialized() and world_size > 1:
            # 创建tensor并进行all_reduce
            train_loss_tensor = torch.tensor(train_loss_epoch, device=device, dtype=torch.float32)
            num_batches_tensor = torch.tensor(num_batches, device=device, dtype=torch.float32)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            
            train_loss_epoch = train_loss_tensor.item()
            num_batches = int(num_batches_tensor.item())
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="验证", leave=False, disable=not is_main_process)
            for H_input, theta_label, phi_label, H_complex in val_progress:
                try:
                    H_input = H_input.to(device, non_blocking=True)
                    theta_label = theta_label.to(device, non_blocking=True)
                    phi_label = phi_label.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        predictions = model(H_input)
                        loss, _, _ = criterion(predictions, (theta_label, phi_label))
                        
                        # 对分布式训练进行loss缩放
                        if use_distributed and world_size > 1:
                            loss = loss / world_size
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    if is_main_process:
                        print(f"验证批次错误: {e}")
                    continue
        
        # 同步验证损失
        if use_distributed and dist.is_initialized() and world_size > 1:
            val_loss_tensor = torch.tensor(val_loss_epoch, device=device, dtype=torch.float32)
            val_batches_tensor = torch.tensor(val_batches, device=device, dtype=torch.float32)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
            
            val_loss_epoch = val_loss_tensor.item()
            val_batches = int(val_batches_tensor.item())
        
        # 计算平均损失
        train_loss_avg = train_loss_epoch / max(num_batches, 1)
        val_loss_avg = val_loss_epoch / max(val_batches, 1)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # 打印结果
        if is_main_process:
            print(f"Epoch {epoch+1:3d}: Train={train_loss_avg:.6f}, Val={val_loss_avg:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # 早停和模型保存 (只在主进程)
        improvement = best_val_loss - val_loss_avg
        if improvement > min_improvement:
            best_val_loss = val_loss_avg
            patience_counter = 0
            if is_main_process:
                # 保存模型时需要处理DDP和DataParallel包装
                if hasattr(model, 'module'):
                    # DDP或DataParallel包装的模型
                    model_to_save = model.module
                else:
                    # 普通模型
                    model_to_save = model
                    
                torch.save(model_to_save.state_dict(), 'best_optimized_model.pth')
                print(f"✓ 保存最佳模型 (Val Loss: {val_loss_avg:.6f}, 改进: {improvement:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            if is_main_process:
                print(f"早停触发！停止训练")
            break
    
    return train_losses, val_losses

def test_optimized_model(model, test_loader, device='cuda'):
    """测试优化模型"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    angle_errors = []
    
    with torch.no_grad():
        for H_input, theta_label, phi_label, H_complex in tqdm(test_loader, desc="测试模型"):
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(H_input)
                
                # 转换为度数
                pred_angles = predictions.cpu().numpy()
                pred_theta_deg = pred_angles[:, 0] * 90
                pred_phi_deg = pred_angles[:, 1] * 180
                
                # 目标角度
                target_theta_deg = theta_label[:, -1].cpu().numpy() * 90
                target_phi_deg = phi_label[:, -1].cpu().numpy() * 180
                
                # 计算误差
                theta_error = np.abs(pred_theta_deg - target_theta_deg)
                phi_error = np.abs(pred_phi_deg - target_phi_deg)
                phi_error = np.minimum(phi_error, 360 - phi_error)
                
                total_error = theta_error + phi_error
                angle_errors.extend(total_error)
                
                for i in range(len(pred_angles)):
                    all_predictions.append({
                        'predicted_theta': pred_theta_deg[i],
                        'predicted_phi': pred_phi_deg[i],
                        'target_theta': target_theta_deg[i],
                        'target_phi': target_phi_deg[i],
                        'theta_error': theta_error[i],
                        'phi_error': phi_error[i],
                        'total_error': total_error[i]
                    })
                    
            except Exception as e:
                print(f"测试批次错误: {e}")
                continue
    
    # 计算统计指标
    if len(angle_errors) > 0:
        mean_error = np.mean(angle_errors)
        std_error = np.std(angle_errors)
        median_error = np.median(angle_errors)
        
        print(f"\n📊 优化模型测试结果:")
        print(f"平均角度误差: {mean_error:.3f}°")
        print(f"角度误差标准差: {std_error:.3f}°")
        print(f"角度误差中位数: {median_error:.3f}°")
        print(f"有效预测数量: {len(all_predictions)}")
    
    return all_predictions

def main():
    """主函数"""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 检查是否使用DataParallel模式
    use_dataparallel = os.environ.get('USE_DATAPARALLEL', '0') == '1'
    
    if use_dataparallel:
        # DataParallel模式：单进程多GPU
        print("使用 DataParallel 模式")
        world_size, rank, local_rank, already_handled = 1, 0, 0, False
        use_distributed = False
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = 'cuda:0'
            print(f"DataParallel 将使用 {torch.cuda.device_count()} 个GPU")
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        # 设置分布式训练
        world_size, rank, local_rank, already_handled = setup_distributed_training()
        if already_handled:
            return  # 如果已经通过multiprocessing处理了，直接返回
        
        use_distributed = world_size > 1 and dist.is_initialized()
        
        # 设备配置
        if torch.cuda.is_available():
            if local_rank is not None:
                device = f'cuda:{local_rank}'
                torch.cuda.set_device(local_rank)
            else:
                device = 'cuda:0'
        else:
            device = 'cpu'
    
    if rank == 0:
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} 内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # 数据路径
    channel_folder = "samples_data/channel_data_opt_20250702_143327"
    angle_folder = "samples_data/angle_data_opt_20250702_143327"
    
    if rank == 0:
        print("正在加载数据集...")
    
    # 创建数据集
    dataset = BeamformingDataset(channel_folder, angle_folder, use_augmentation=True)
    
    # 数据集划分
    total_samples = len(dataset)
    train_size = 800
    test_size = total_samples - train_size
    
    indices = list(range(total_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    val_size = int(train_size * 0.20)
    train_final_indices = train_indices[val_size:]
    val_indices = train_indices[:val_size]
    
    if rank == 0:
        print(f"优化模型数据集划分:")
        print(f"  训练集: {len(train_final_indices)} 样本")
        print(f"  验证集: {len(val_indices)} 样本") 
        print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器
    batch_size = 2  # 基础批次大小
    
    # DataParallel模式下可以增加批次大小
    if use_dataparallel and torch.cuda.device_count() > 1:
        batch_size = 2 * torch.cuda.device_count()
        print(f"DataParallel模式，批次大小调整为: {batch_size}")
    
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    # 只有真正的分布式训练才使用DistributedSampler
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=2, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=test_sampler,
        num_workers=2, 
        pin_memory=True
    )
    
    # 创建优化的模型
    model = OptimizedTransformerBeamformingNet(
        input_dim=256*256*2,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"优化模型参数数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 将模型移到设备并设置并行模式
    model.to(device)
    
    if use_dataparallel and torch.cuda.device_count() > 1:
        print(f"使用 DataParallel 包装模型，GPU数量: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
        use_distributed_training_flag = False
    elif use_distributed:
        use_distributed_training_flag = True
    else:
        use_distributed_training_flag = False
    
    # 训练模型
    if rank == 0:
        print(f"\n🚀 开始优化模型训练 ({'DataParallel' if use_dataparallel else 'DDP' if use_distributed else 'Single GPU'})...")
    
    train_losses, val_losses = train_optimized_model(
        model, train_loader, val_loader, 
        num_epochs=150,
        lr=1e-4,
        device=device,
        use_distributed=use_distributed_training_flag,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    
    # 只在主进程进行测试和可视化
    if rank == 0:
        # 加载最佳模型
        if os.path.exists('best_optimized_model.pth'):
            # 处理DataParallel保存的模型
            checkpoint = torch.load('best_optimized_model.pth')
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print("✓ 加载最佳优化模型完成")
        
        # 测试模型
        print("\n🧪 开始优化模型测试...")
        predictions = test_optimized_model(model, test_loader, device=device)
        
        # 绘制结果
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 训练损失曲线
        axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
        axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Optimized Model Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        if len(predictions) > 0:
            pred_theta = [p['predicted_theta'] for p in predictions]
            target_theta = [p['target_theta'] for p in predictions]
            pred_phi = [p['predicted_phi'] for p in predictions]
            target_phi = [p['target_phi'] for p in predictions]
            
            # θ角度预测
            axes[0, 1].scatter(target_theta, pred_theta, alpha=0.6, s=25)
            min_theta = min(min(target_theta), min(pred_theta))
            max_theta = max(max(target_theta), max(pred_theta))
            axes[0, 1].plot([min_theta, max_theta], [min_theta, max_theta], 'r--', label='Perfect')
            axes[0, 1].set_xlabel('True θ (degrees)')
            axes[0, 1].set_ylabel('Predicted θ (degrees)')
            axes[0, 1].set_title('θ Angle Prediction (Optimized)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # φ角度预测
            axes[1, 0].scatter(target_phi, pred_phi, alpha=0.6, s=25)
            min_phi = min(min(target_phi), min(pred_phi))
            max_phi = max(max(target_phi), max(pred_phi))
            axes[1, 0].plot([min_phi, max_phi], [min_phi, max_phi], 'r--', label='Perfect')
            axes[1, 0].set_xlabel('True φ (degrees)')
            axes[1, 0].set_ylabel('Predicted φ (degrees)')
            axes[1, 0].set_title('φ Angle Prediction (Optimized)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 误差分布
            errors = [p['total_error'] for p in predictions]
            axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--',
                              label=f'Mean: {np.mean(errors):.2f}°')
            axes[1, 1].set_xlabel('Total Angle Error (degrees)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Error Distribution (Optimized)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimized_training_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📈 优化模型训练结果图表已保存")
        
        # 打印总结
        print(f"\n📋 优化模型训练总结:")
        if train_losses:
            print(f"• 最终训练损失: {train_losses[-1]:.6f}")
            print(f"• 最终验证损失: {val_losses[-1]:.6f}")
            print(f"• 总训练轮数: {len(train_losses)}")
            print(f"• 模型规模: {total_params/1e6:.1f}M 参数")
            
            training_mode = "DataParallel" if use_dataparallel else "DDP" if use_distributed else "Single GPU"
            print(f"• 训练模式: {training_mode}")
    
    # 清理分布式环境
    if use_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()