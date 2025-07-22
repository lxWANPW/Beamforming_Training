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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import os

class BeamformingDataset(Dataset):
    """修复版天线阵列信道数据集"""
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
        """数据增强策略"""
        if not self.use_augmentation:
            return H_real, H_imag, theta_rad, phi_rad
            
        # 1. 噪声增强
        noise_std = 0.01
        H_real = H_real + np.random.normal(0, noise_std, H_real.shape)
        H_imag = H_imag + np.random.normal(0, noise_std, H_imag.shape)
        
        # 2. 旋转增强 (小角度旋转)
        angle_noise_std = 0.02  # 约1.1度的标准差
        theta_rad = theta_rad + np.random.normal(0, angle_noise_std, theta_rad.shape)
        phi_rad = phi_rad + np.random.normal(0, angle_noise_std, phi_rad.shape)
        
        # 3. 幅度缩放
        scale_factor = np.random.uniform(0.95, 1.05)
        H_real = H_real * scale_factor
        H_imag = H_imag * scale_factor
        
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
        
        # 数据增强
        H_real, H_imag, theta_rad, phi_rad = self._add_data_augmentation(
            H_real, H_imag, theta_rad, phi_rad
        )
        
        # 组合实部和虚部
        H_complex = H_real + 1j * H_imag
        
        # 改进的数据归一化（避免极值）
        H_real_std = np.std(H_real)
        H_imag_std = np.std(H_imag)
        
        if H_real_std > 1e-8:
            H_real_norm = (H_real - np.mean(H_real)) / H_real_std
        else:
            H_real_norm = H_real
            
        if H_imag_std > 1e-8:
            H_imag_norm = (H_imag - np.mean(H_imag)) / H_imag_std
        else:
            H_imag_norm = H_imag
        
        # 裁剪极值以避免梯度爆炸
        H_real_norm = np.clip(H_real_norm, -8, 8)
        H_imag_norm = np.clip(H_imag_norm, -8, 8)
        
        # 转换为PyTorch tensor
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)  # [2, 256, 256, 50]
        H_input = torch.FloatTensor(H_input)
        
        # 改进的角度标签归一化
        # 使用更精确的归一化方式
        theta_norm = theta_rad / (np.pi / 2)  # 归一化到[-2, 2]范围，然后裁剪
        phi_norm = phi_rad / np.pi  # 归一化到[-1, 1]
        
        theta_norm = np.clip(theta_norm, -1, 1)
        phi_norm = np.clip(phi_norm, -1, 1)
        
        theta_label = torch.FloatTensor(theta_norm)  # [50]
        phi_label = torch.FloatTensor(phi_norm)      # [50]
        
        return H_input, theta_label, phi_label, H_complex

class AdvancedPositionalEncoding(nn.Module):
    """改进的位置编码，支持2D空间信息"""
    def __init__(self, d_model, max_len=5000, spatial_dims=None):
        super(AdvancedPositionalEncoding, self).__init__()
        
        # 时间位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        # 学习式位置编码
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        
    def forward(self, x):
        # 组合固定和学习式位置编码
        return x + self.pe[:x.size(0), :] + self.learned_pe[:x.size(0), :]

class SpatialAttention(nn.Module):
    """空间注意力机制，关注重要的空间特征"""
    def __init__(self, d_model):
        super(SpatialAttention, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(),
            nn.Conv2d(d_model // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [batch, time, d_model]
        # 需要重塑为空间维度进行空间注意力计算
        B, T, D = x.shape
        sqrt_d = int(math.sqrt(D))
        if sqrt_d * sqrt_d == D:
            # 如果可以完美开方，重塑为2D
            x_2d = x.view(B, T, sqrt_d, sqrt_d)
            x_2d = x_2d.permute(0, 1, 2, 3).contiguous()
            
            # 应用空间注意力到每个时间步
            attention_maps = []
            for t in range(T):
                att_map = self.spatial_conv(x_2d[:, t:t+1])  # [B, 1, sqrt_d, sqrt_d]
                attention_maps.append(att_map)
            
            attention = torch.cat(attention_maps, dim=1)  # [B, T, sqrt_d, sqrt_d]
            attention = attention.view(B, T, D)
            
            return x * attention
        else:
            # 如果不能完美开方，直接返回
            return x

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器，提取不同层次的特征"""
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 分阶段降维，减少参数量
        intermediate_dims = [input_dim, input_dim//4, input_dim//16, d_model*2, d_model]
        
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
        
        # 多尺度分支
        self.multi_scale_branches = nn.ModuleList([
            # 局部特征 (小感受野)
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model//8),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ),
            # 中等感受野
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model//8),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ),
            # 全局特征 (大感受野)
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model//8),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * 4, d_model, kernel_size=1),  # 3个分支 + 原始特征
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 逐步降维
        for layer in self.feature_layers:
            x = layer(x)
        
        # 多尺度特征提取
        original_features = x
        multi_scale_features = []
        for branch in self.multi_scale_branches:
            multi_scale_features.append(branch(x))
        
        # 融合所有特征
        all_features = torch.cat([original_features] + multi_scale_features, dim=1)
        fused_features = self.fusion(all_features)
        
        return fused_features

class AdvancedTransformerBeamformingNet(nn.Module):
    """高级Transformer波束成形网络，显著提升角度预测精度 - 内存优化版本"""
    def __init__(self, input_dim=256*256*2, d_model=768, nhead=12, num_layers=8, dropout=0.1):
        super(AdvancedTransformerBeamformingNet, self).__init__()
        
        self.d_model = d_model
        
        # 优化的多尺度特征提取器（减少参数）
        self.feature_extractor = MultiScaleFeatureExtractor(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout
        )
        
        # 改进的位置编码
        self.pos_encoder = AdvancedPositionalEncoding(d_model, max_len=100)
        
        # 空间注意力（简化版）
        self.spatial_attention = SpatialAttention(d_model)
        
        # 优化的Transformer编码器层（减少层数和FFN大小）
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 减少FFN大小
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 单级注意力聚合（减少内存使用）
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 优化的角度预测头
        self.shared_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Theta预测分支
        self.theta_predictor = nn.Sequential(
            nn.Linear(d_model // 4, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Phi预测分支
        self.phi_predictor = nn.Sequential(
            nn.Linear(d_model // 4, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 可学习的查询向量
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.8)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, features]
        
        # 多尺度特征提取
        x = x.permute(0, 2, 1)  # [batch, features, time_steps]
        x = self.feature_extractor(x)  # [batch, d_model, time_steps]
        x = x.permute(0, 2, 1)  # [batch, time_steps, d_model]
        
        # 空间注意力
        x = self.spatial_attention(x)
        
        # 位置编码和缩放
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [time_steps, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, time_steps, d_model]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [batch, time_steps, d_model]
        
        # 注意力聚合
        query = self.query_vector.expand(batch_size, -1, -1)
        aggregated, _ = self.attention_pooling(query, encoded, encoded)
        
        final_features = aggregated.squeeze(1)  # [batch, d_model]
        
        # 共享特征提取
        shared_features = self.shared_predictor(final_features)
        
        # 分离预测
        theta_pred = self.theta_predictor(shared_features)  # [batch, 1]
        phi_pred = self.phi_predictor(shared_features)      # [batch, 1]
        
        # 组合输出
        angles = torch.cat([theta_pred, phi_pred], dim=1)  # [batch, 2]
        
        return angles

class AdvancedLossFunction(nn.Module):
    """高级损失函数，结合多种损失策略"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        super(AdvancedLossFunction, self).__init__()
        self.alpha = alpha  # Focal loss权重
        self.beta = beta    # Huber loss权重
        self.gamma = gamma  # Contrastive learning权重
        
    def focal_huber_loss(self, pred, target, delta=0.5):
        """Focal Huber损失"""
        error = torch.abs(pred - target)
        focal_weight = torch.pow(error + 1e-8, 0.3)  # 更温和的权重
        huber_loss = F.huber_loss(pred, target, delta=delta, reduction='none')
        return (focal_weight * huber_loss).mean()
    
    def angle_consistency_loss(self, pred_theta, pred_phi, target_theta, target_phi):
        """角度一致性损失，考虑角度的周期性"""
        # Theta损失
        theta_diff = pred_theta - target_theta
        theta_loss = torch.min(torch.abs(theta_diff), 
                              torch.abs(theta_diff + 2), 
                              torch.abs(theta_diff - 2))
        
        # Phi损失
        phi_diff = pred_phi - target_phi
        phi_loss = torch.min(torch.abs(phi_diff), 
                            torch.abs(phi_diff + 2), 
                            torch.abs(phi_diff - 2))
        
        return theta_loss.mean() + phi_loss.mean()
    
    def smoothness_loss(self, predictions):
        """平滑性损失，减少预测的突变"""
        # 对于batch内的预测值，计算相邻样本的差异
        if predictions.size(0) > 1:
            diff = torch.diff(predictions, dim=0)
            return torch.mean(torch.abs(diff))
        return torch.tensor(0.0, device=predictions.device)
    
    def forward(self, predictions, targets):
        pred_angles = predictions  # [batch, 2]
        target_theta, target_phi = targets
        
        # 使用最后一个时间步的目标
        target_theta_last = target_theta[:, -1]  # [batch]
        target_phi_last = target_phi[:, -1]      # [batch]
        
        # 主要损失：Focal Huber
        theta_loss = self.focal_huber_loss(pred_angles[:, 0], target_theta_last)
        phi_loss = self.focal_huber_loss(pred_angles[:, 1], target_phi_last)
        main_loss = self.alpha * (theta_loss + phi_loss)
        
        # 角度一致性损失
        consistency_loss = self.beta * self.angle_consistency_loss(
            pred_angles[:, 0], pred_angles[:, 1], 
            target_theta_last, target_phi_last
        )
        
        # 平滑性损失
        smooth_loss = self.gamma * self.smoothness_loss(pred_angles)
        
        total_loss = main_loss + consistency_loss + smooth_loss
        
        # 防止损失爆炸
        total_loss = torch.clamp(total_loss, 0, 100)
        
        return total_loss, theta_loss, phi_loss

def setup_distributed():
    """设置分布式训练环境"""
    if 'WORLD_SIZE' in os.environ:
        # 从环境变量获取分布式信息
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        print(f"初始化分布式训练: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        
        # 设置CUDA设备
        torch.cuda.set_device(local_rank)
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            print(f"Rank {rank}: 分布式进程组初始化完成")
        
        return world_size, rank, local_rank
    else:
        # 非分布式环境
        return 1, 0, 0

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_advanced_model(model, train_loader, val_loader, num_epochs=200, lr=2e-4, device='cuda', use_distributed=False):
    """高级模型训练函数 - 多GPU支持版本"""
    # 设置分布式训练
    world_size, rank, local_rank = 1, 0, 0
    if use_distributed:
        world_size, rank, local_rank = setup_distributed()
        if dist.is_initialized():
            device = f'cuda:{local_rank}'
        else:
            print("分布式初始化失败，回退到单GPU训练")
            use_distributed = False
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    model.to(device)
    
    # 包装为分布式模型
    if use_distributed and dist.is_initialized() and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print(f"Rank {rank}: 模型已包装为DDP")
    
    # 使用混合精度训练
    scaler = GradScaler()
    
    # 优化器设置 - 针对分布式训练调整学习率
    effective_lr = lr * world_size if use_distributed and dist.is_initialized() and world_size > 1 else lr
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=effective_lr, 
        weight_decay=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度
    warmup_epochs = 20
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=effective_lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,
        div_factor=20,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    # 高级损失函数
    criterion = AdvancedLossFunction(alpha=1.0, beta=0.5, gamma=0.2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    min_improvement = 5e-6
    
    # 梯度累积设置
    accumulation_steps = 2  # 每2个batch累积一次梯度
    
    # 只在主进程打印训练信息
    is_main_process = rank == 0
    if is_main_process:
        print(f"开始高级模型训练，学习率: {effective_lr}, 梯度累积步数: {accumulation_steps}")
        if use_distributed and dist.is_initialized() and world_size > 1:
            print(f"使用 {world_size} 个GPU进行分布式训练")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_theta_loss = 0
        train_phi_loss = 0
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}", leave=False, disable=not is_main_process)
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 检查输入数据的有效性
                if torch.isnan(H_input).any() or torch.isinf(H_input).any():
                    continue
                
                # 混合精度前向传播
                with torch.amp.autocast('cuda'):
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = criterion(predictions, (theta_label, phi_label))
                    loss = loss / accumulation_steps  # 归一化损失
                
                # 检查损失的有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss_epoch += loss.item() * accumulation_steps
                train_theta_loss += theta_loss.item()
                train_phi_loss += phi_loss.item()
                num_batches += 1
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                train_progress.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.5f}',
                    'θ': f'{theta_loss.item():.5f}',
                    'φ': f'{phi_loss.item():.5f}',
                    'LR': f'{current_lr:.2e}'
                })
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if is_main_process:
                    print(f"训练批次 {batch_idx} 出错: {e}")
                # 清理缓存并继续
                torch.cuda.empty_cache()
                continue
        
        # 同步所有进程的梯度统计
        if use_distributed and dist.is_initialized() and world_size > 1:
            # 收集所有进程的损失
            train_loss_tensor = torch.tensor(train_loss_epoch, device=device)
            train_theta_tensor = torch.tensor(train_theta_loss, device=device)
            train_phi_tensor = torch.tensor(train_phi_loss, device=device)
            batch_count_tensor = torch.tensor(num_batches, device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_theta_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_phi_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
            
            train_loss_epoch = train_loss_tensor.item()
            train_theta_loss = train_theta_tensor.item()
            train_phi_loss = train_phi_tensor.item()
            num_batches = batch_count_tensor.item()
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_theta_loss = 0
        val_phi_loss = 0
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
                        loss, theta_loss, phi_loss = criterion(predictions, (theta_label, phi_label))
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_theta_loss += theta_loss.item()
                        val_phi_loss += phi_loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    continue
            
            # 同步验证损失
            if use_distributed and dist.is_initialized() and world_size > 1:
                val_loss_tensor = torch.tensor(val_loss_epoch, device=device)
                val_theta_tensor = torch.tensor(val_theta_loss, device=device)
                val_phi_tensor = torch.tensor(val_phi_loss, device=device)
                val_batch_tensor = torch.tensor(val_batches, device=device)
                
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_theta_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_phi_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_batch_tensor, op=dist.ReduceOp.SUM)
                
                val_loss_epoch = val_loss_tensor.item()
                val_theta_loss = val_theta_tensor.item()
                val_phi_loss = val_phi_tensor.item()
                val_batches = val_batch_tensor.item()
            
            # 清理验证阶段缓存
            torch.cuda.empty_cache()
        
        # 计算平均损失
        if num_batches > 0:
            train_loss_avg = train_loss_epoch / num_batches
            train_theta_avg = train_theta_loss / num_batches
            train_phi_avg = train_phi_loss / num_batches
        else:
            train_loss_avg = float('inf')
            train_theta_avg = float('inf')
            train_phi_avg = float('inf')
            
        if val_batches > 0:
            val_loss_avg = val_loss_epoch / val_batches
            val_theta_avg = val_theta_loss / val_batches
            val_phi_avg = val_phi_loss / val_batches
        else:
            val_loss_avg = float('inf')
            val_theta_avg = float('inf')
            val_phi_avg = float('inf')
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # 动态早停策略
        current_early_stop_patience = early_stop_patience
        if val_loss_avg < 0.0005:  # 更严格的阈值
            current_early_stop_patience = 80
        
        # 打印epoch结果 (只在主进程)
        current_lr = optimizer.param_groups[0]['lr']
        train_val_gap = val_loss_avg - train_loss_avg
        if is_main_process:
            print(f"Epoch {epoch+1:3d}: Train={train_loss_avg:.7f} (θ:{train_theta_avg:.7f}, φ:{train_phi_avg:.7f}), "
                  f"Val={val_loss_avg:.7f} (θ:{val_theta_avg:.7f}, φ:{val_phi_avg:.7f}), "
                  f"Gap={train_val_gap:.7f}, LR={current_lr:.2e}, Patience={patience_counter}/{current_early_stop_patience}")
        
        # 早停和模型保存 (只在主进程)
        improvement = best_val_loss - val_loss_avg
        if improvement > min_improvement:
            best_val_loss = val_loss_avg
            patience_counter = 0
            if is_main_process:
                # 保存模型时需要处理DDP包装
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), 'best_advanced_beamforming_model.pth')
                print(f"✓ 保存最佳高级模型 (Val Loss: {val_loss_avg:.7f}, 改进: {improvement:.7f})")
        else:
            patience_counter += 1
            
        if patience_counter >= current_early_stop_patience:
            if is_main_process:
                print(f"早停触发！在epoch {epoch+1}停止训练 (patience: {patience_counter}/{current_early_stop_patience})")
            break
    
    # 清理分布式环境
    if use_distributed and dist.is_initialized():
        cleanup_distributed()
    
    return train_losses, val_losses

def test_advanced_model(model, test_loader, device='cuda'):
    """测试高级模型性能"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    angle_errors = []
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="高级模型测试")
        
        for H_input, theta_label, phi_label, H_complex in test_progress:
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                
                # 预测
                with torch.amp.autocast('cuda'):
                    predictions = model(H_input)
                
                # 转换为度数（反归一化）
                pred_angles = predictions.cpu().numpy()
                pred_theta_norm = pred_angles[:, 0]  # [-1, 1]
                pred_phi_norm = pred_angles[:, 1]    # [-1, 1]
                
                # 反归一化到度数
                pred_theta_deg = pred_theta_norm * 90
                pred_phi_deg = pred_phi_norm * 180
                
                # 目标角度（使用最后时间步）
                target_theta_norm = theta_label[:, -1].cpu().numpy()
                target_phi_norm = phi_label[:, -1].cpu().numpy()
                
                target_theta_deg = target_theta_norm * 90
                target_phi_deg = target_phi_norm * 180
                
                # 计算角度误差
                theta_error = np.abs(pred_theta_deg - target_theta_deg)
                phi_error = np.abs(pred_phi_deg - target_phi_deg)
                
                # 处理phi角度的周期性
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
        
        print(f"\n📊 高级模型测试结果:")
        print(f"平均角度误差: {mean_error:.3f}°")
        print(f"角度误差标准差: {std_error:.3f}°")
        print(f"角度误差中位数: {median_error:.3f}°")
        print(f"有效预测数量: {len(all_predictions)}")
        
        if len(all_predictions) > 0:
            theta_errors = [p['theta_error'] for p in all_predictions]
            phi_errors = [p['phi_error'] for p in all_predictions]
            print(f"θ误差: {np.mean(theta_errors):.3f}° ± {np.std(theta_errors):.3f}°")
            print(f"φ误差: {np.mean(phi_errors):.3f}° ± {np.std(phi_errors):.3f}°")
            
            # 误差分布统计
            errors_below_1 = sum(1 for e in angle_errors if e < 1.0)
            errors_below_2 = sum(1 for e in angle_errors if e < 2.0)
            errors_below_5 = sum(1 for e in angle_errors if e < 5.0)
            
            total = len(angle_errors)
            print(f"误差 < 1°: {errors_below_1}/{total} ({errors_below_1/total*100:.1f}%)")
            print(f"误差 < 2°: {errors_below_2}/{total} ({errors_below_2/total*100:.1f}%)")
            print(f"误差 < 5°: {errors_below_5}/{total} ({errors_below_5/total*100:.1f}%)")
    else:
        print("没有有效的测试结果")
    
    return all_predictions

def main():
    """主函数"""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 检测GPU环境和分布式设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    
    # 检查是否在分布式环境中运行
    is_distributed_env = 'WORLD_SIZE' in os.environ
    use_distributed = is_distributed_env and gpu_count > 1
    
    print(f"使用设备: {device}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {gpu_count}")
    print(f"分布式环境: {is_distributed_env}")
    print(f"使用分布式训练: {use_distributed}")
    
    if torch.cuda.is_available():
        print(f"主GPU型号: {torch.cuda.get_device_name()}")
        print(f"主GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        if use_distributed:
            print(f"将使用分布式训练")
        elif gpu_count > 1:
            print(f"检测到 {gpu_count} 个GPU，但未在分布式环境中运行，将使用单GPU训练")
            print("提示: 使用 'python -m torch.distributed.launch --nproc_per_node={} advanced_beamforming_model.py' 启动分布式训练".format(gpu_count))
    
    # 数据路径
    channel_folder = "samples_data/channel_data_opt_20250702_143327"
    angle_folder = "samples_data/angle_data_opt_20250702_143327"
    
    print("正在加载数据集...")
    
    # 创建数据集（开启数据增强）
    dataset = BeamformingDataset(channel_folder, angle_folder, use_augmentation=True)
    
    # 按要求划分数据集
    total_samples = len(dataset)
    train_size = 800
    test_size = total_samples - train_size
    
    indices = list(range(total_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 从训练集中划分验证集
    val_size = int(train_size * 0.20)  # 20%作为验证集
    train_final_indices = train_indices[val_size:]
    val_indices = train_indices[:val_size]
    
    print(f"高级模型数据集划分:")
    print(f"  训练集: {len(train_final_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本") 
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器 - 根据是否分布式来决定sampler
    batch_size = 4  # 减小批次大小适配更大模型
    
    # 只在真正的分布式环境中创建DistributedSampler
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if use_distributed:
        # 初始化分布式环境
        world_size, rank, local_rank = setup_distributed()
        if dist.is_initialized():
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            print(f"Rank {rank}: 创建分布式sampler")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=test_sampler,
        num_workers=4, 
        pin_memory=True
    )
    
    # 创建高级Transformer模型
    model = AdvancedTransformerBeamformingNet(
        input_dim=256*256*2,
        d_model=1024,     # 更大的模型维度
        nhead=16,         # 更多注意力头
        num_layers=12,    # 更深的网络
        dropout=0.1       # 适度的dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"高级模型参数数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 训练模型
    print("\n🚀 开始高级模型训练...")
    train_losses, val_losses = train_advanced_model(
        model, train_loader, val_loader, 
        num_epochs=200,
        lr=2e-4,
        device=device,
        use_distributed=use_distributed
    )
    
    # 加载最佳模型
    if os.path.exists('best_advanced_beamforming_model.pth'):
        model.load_state_dict(torch.load('best_advanced_beamforming_model.pth'))
        print("✓ 加载最佳高级模型完成")
    
    # 测试模型
    print("\n🧪 开始高级模型测试...")
    predictions = test_advanced_model(model, test_loader, device=device)
    
    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 训练损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8, linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Advanced Model Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    if len(predictions) > 0:
        pred_theta = [p['predicted_theta'] for p in predictions]
        target_theta = [p['target_theta'] for p in predictions]
        pred_phi = [p['predicted_phi'] for p in predictions]
        target_phi = [p['target_phi'] for p in predictions]
        
        # θ角度预测
        axes[0, 1].scatter(target_theta, pred_theta, alpha=0.6, s=25, c='blue')
        min_theta = min(min(target_theta), min(pred_theta))
        max_theta = max(max(target_theta), max(pred_theta))
        axes[0, 1].plot([min_theta, max_theta], [min_theta, max_theta], 'r--', label='Perfect', linewidth=2)
        axes[0, 1].set_xlabel('True θ (degrees)')
        axes[0, 1].set_ylabel('Predicted θ (degrees)')
        axes[0, 1].set_title('θ Angle Prediction (Advanced)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # φ角度预测
        axes[0, 2].scatter(target_phi, pred_phi, alpha=0.6, s=25, c='green')
        min_phi = min(min(target_phi), min(pred_phi))
        max_phi = max(max(target_phi), max(pred_phi))
        axes[0, 2].plot([min_phi, max_phi], [min_phi, max_phi], 'r--', label='Perfect', linewidth=2)
        axes[0, 2].set_xlabel('True φ (degrees)')
        axes[0, 2].set_ylabel('Predicted φ (degrees)')
        axes[0, 2].set_title('φ Angle Prediction (Advanced)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 误差分布
        errors = [p['total_error'] for p in predictions]
        axes[1, 0].hist(errors, bins=40, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(errors):.2f}°')
        axes[1, 0].axvline(np.median(errors), color='blue', linestyle='--', linewidth=2,
                          label=f'Median: {np.median(errors):.2f}°')
        axes[1, 0].set_xlabel('Total Angle Error (degrees)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution (Advanced)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # θ和φ误差对比
        theta_errors = [p['theta_error'] for p in predictions]
        phi_errors = [p['phi_error'] for p in predictions]
        
        x_pos = [1, 2]
        error_means = [np.mean(theta_errors), np.mean(phi_errors)]
        error_stds = [np.std(theta_errors), np.std(phi_errors)]
        
        axes[1, 1].bar(x_pos, error_means, yerr=error_stds, capsize=5, 
                      color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['θ Error', 'φ Error'])
        axes[1, 1].set_ylabel('Mean Error (degrees)')
        axes[1, 1].set_title('θ vs φ Error Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 累积误差分布
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 2].plot(sorted_errors, cumulative_prob * 100, linewidth=2, color='purple')
        axes[1, 2].axvline(np.percentile(sorted_errors, 90), color='red', linestyle='--',
                          label=f'90%: {np.percentile(sorted_errors, 90):.2f}°')
        axes[1, 2].axvline(np.percentile(sorted_errors, 95), color='orange', linestyle='--',
                          label=f'95%: {np.percentile(sorted_errors, 95):.2f}°')
        axes[1, 2].set_xlabel('Total Angle Error (degrees)')
        axes[1, 2].set_ylabel('Cumulative Probability (%)')
        axes[1, 2].set_title('Cumulative Error Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 高级模型训练结果图表已保存为 'advanced_training_results.png'")
    
    # 打印训练总结
    print(f"\n📋 高级模型训练总结:")
    if train_losses:
        print(f"• 最终训练损失: {train_losses[-1]:.6f}")
        print(f"• 最终验证损失: {val_losses[-1]:.6f}")
        print(f"• 总训练轮数: {len(train_losses)}")
        print(f"• 损失改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
        print(f"• 模型规模: {total_params/1e6:.1f}M 参数")
    
    return model, predictions

if __name__ == "__main__":
    main()
