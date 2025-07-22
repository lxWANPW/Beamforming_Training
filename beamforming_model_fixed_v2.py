import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
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

class BeamformingDataset(Dataset):
    """修复版天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        
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
        H_real_norm = np.clip(H_real_norm, -10, 10)
        H_imag_norm = np.clip(H_imag_norm, -10, 10)
        
        # 转换为PyTorch tensor
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)  # [2, 256, 256, 50]
        H_input = torch.FloatTensor(H_input)
        
        # 改进的角度标签归一化
        theta_rad = self.theta_array[idx] * np.pi / 180  # 转换为弧度
        phi_rad = self.phi_array[idx] * np.pi / 180
        
        # 使用更稳定的归一化方式
        theta_norm = theta_rad / (np.pi / 2)  # 归一化到[-2, 2]范围，然后裁剪
        phi_norm = phi_rad / np.pi  # 归一化到[-1, 1]
        
        theta_norm = np.clip(theta_norm, -1, 1)
        phi_norm = np.clip(phi_norm, -1, 1)
        
        theta_label = torch.FloatTensor(theta_norm)  # [50]
        phi_label = torch.FloatTensor(phi_norm)      # [50]
        
        return H_input, theta_label, phi_label, H_complex

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器，增强空间-时间特征表达"""
    def __init__(self, input_channels=2, d_model=1024):
        super().__init__()
        
        # 3D卷积分支 - 捕获空间-时间相关性
        self.conv3d_branch = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d((2,2,1)),  # 空间降采样，保持时间维度
            
            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, 512, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 50)),  # 全局空间池化
        )
        
        # 多尺度2D卷积分支
        self.multiscale_2d = nn.ModuleList([
            # 小尺度特征
            nn.Sequential(
                nn.Conv2d(input_channels, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            ),
            # 中尺度特征
            nn.Sequential(
                nn.Conv2d(input_channels, 128, 5, padding=2),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, 5, padding=2),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            ),
            # 大尺度特征
            nn.Sequential(
                nn.Conv2d(input_channels, 128, 7, padding=3),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, 256, 7, padding=3),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
        ])
        
        # 特征融合网络
        total_features = 512 + 256 * 3  # 3D特征 + 3个2D分支
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
    def forward(self, x):
        batch, channels, height, width, time = x.shape
        
        # 3D卷积分支
        conv3d_out = self.conv3d_branch(x)  # [batch, 512, 1, 1, 50]
        conv3d_features = conv3d_out.squeeze(2).squeeze(2)  # [batch, 512, 50]
        
        # 多尺度2D卷积分支
        multiscale_features = []
        for t in range(time):
            frame = x[:, :, :, :, t]  # [batch, 2, 256, 256]
            frame_features = []
            for conv2d in self.multiscale_2d:
                feature = conv2d(frame).squeeze(-1).squeeze(-1)  # [batch, 256]
                frame_features.append(feature)
            multiscale_features.append(torch.cat(frame_features, dim=-1))  # [batch, 768]
        
        multiscale_features = torch.stack(multiscale_features, dim=1)  # [batch, 50, 768]
        
        # 特征融合
        combined_features = []
        for t in range(time):
            conv3d_t = conv3d_features[:, :, t]  # [batch, 512]
            multiscale_t = multiscale_features[:, t, :]  # [batch, 768]
            combined_t = torch.cat([conv3d_t, multiscale_t], dim=-1)  # [batch, 1280]
            fused_t = self.feature_fusion(combined_t)  # [batch, d_model]
            combined_features.append(fused_t)
        
        return torch.stack(combined_features, dim=1)  # [batch, 50, d_model]

class EnhancedTransformerBlock(nn.Module):
    """增强的Transformer块，加入更多非线性和残差连接"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # 增强的前馈网络
        self.enhanced_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        
        # 门控线性单元
        self.glu = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-LN自注意力
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)
        
        # Pre-LN前馈网络
        residual = x
        x = self.norm2(x)
        ffn_out = self.enhanced_ffn(x)
        x = residual + self.dropout(ffn_out)
        
        # 门控线性单元
        residual = x
        x = self.norm3(x)
        glu_out = self.glu(x)
        x = residual + self.dropout(glu_out)
        
        return x

class HierarchicalAttentionPooling(nn.Module):
    """分层注意力池化，从局部到全局聚合信息"""
    def __init__(self, d_model, nhead=8):
        super().__init__()
        
        # 局部注意力（相邻时间步）
        self.local_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # 全局注意力
        self.global_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # 可学习查询向量
        self.local_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 局部注意力聚合
        local_query = self.local_query.expand(batch_size, -1, -1)
        local_out, _ = self.local_attn(local_query, x, x)
        
        # 全局注意力聚合
        global_query = self.global_query.expand(batch_size, -1, -1)
        global_out, _ = self.global_attn(global_query, x, x)
        
        # 融合局部和全局特征
        combined = torch.cat([local_out.squeeze(1), global_out.squeeze(1)], dim=-1)
        fused = self.fusion(combined)
        
        return fused

class AdvancedAnglePredictor(nn.Module):
    """高级角度预测器，包含多个预测头和集成"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        
        # 深度特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 多个专门的预测头
        self.theta_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.phi_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # 联合预测头（用于集成）
        self.joint_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        
        # 集成权重
        self.ensemble_weights = nn.Parameter(torch.ones(2))
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 独立预测
        theta_pred = self.theta_predictor(features)
        phi_pred = self.phi_predictor(features)
        independent_pred = torch.cat([theta_pred, phi_pred], dim=-1)
        
        # 联合预测
        joint_pred = self.joint_predictor(features)
        
        # 集成预测结果
        weights = F.softmax(self.ensemble_weights, dim=0)
        final_pred = weights[0] * independent_pred + weights[1] * joint_pred
        
        return final_pred

class UltraTransformerBeamformingNet(nn.Module):
    """超强Transformer波束成形网络"""
    def __init__(self, d_model=1024, nhead=16, num_layers=12, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 多尺度特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            input_channels=2, d_model=d_model
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # 多个增强Transformer块
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 分层注意力池化
        self.hierarchical_pooling = HierarchicalAttentionPooling(
            d_model=d_model, nhead=nhead
        )
        
        # 高级角度预测器
        self.angle_predictor = AdvancedAnglePredictor(
            d_model=d_model, dropout=dropout
        )
        
        # 权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # 多尺度特征提取
        x = self.feature_extractor(x)  # [batch, 50, d_model]
        
        # 位置编码
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [50, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, 50, d_model]
        
        # 多层增强Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 分层注意力池化
        aggregated = self.hierarchical_pooling(x)  # [batch, d_model]
        
        # 角度预测
        angles = self.angle_predictor(aggregated)  # [batch, 2]
        
        return angles

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBeamformingNet(nn.Module):
    """基于Transformer的波束成形网络，提升角度预测精度"""
    def __init__(self, input_dim=256*256*2, d_model=768, nhead=12, num_layers=8, dropout=0.15):
        super(TransformerBeamformingNet, self).__init__()
        
        self.d_model = d_model
        
        # 高效的特征提取器（使用卷积降维）
        self.feature_extractor = nn.Sequential(
            # 先用卷积降维减少计算量
            nn.Conv1d(input_dim, d_model * 2, kernel_size=1),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # 更强的Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 更大的FFN
            dropout=dropout,
            activation='gelu',  # 使用GELU激活
            batch_first=True,
            norm_first=True  # Pre-LN架构，更稳定
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 多头注意力聚合（替代简单的最后时间步）
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 改进的角度预测头（增加dropout防止过拟合）
        self.angle_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # 更高的dropout
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 输出层前轻微dropout
            nn.Linear(64, 2)  # theta, phi
        )
        
        # 可学习的查询向量用于注意力聚合
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 改进的权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 使用更好的初始化
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, features]
        
        # 特征提取（使用1D卷积）
        x = x.permute(0, 2, 1)  # [batch, features, time_steps]
        x = self.feature_extractor(x)  # [batch, d_model, time_steps]
        x = x.permute(0, 2, 1)  # [batch, time_steps, d_model]
        
        # 输入缩放和位置编码
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [time_steps, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, time_steps, d_model]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [batch, time_steps, d_model]
        
        # 注意力聚合（更好的时序信息整合）
        query = self.query_vector.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        aggregated, attention_weights = self.attention_pooling(
            query, encoded, encoded
        )  # [batch, 1, d_model]
        aggregated = aggregated.squeeze(1)  # [batch, d_model]
        
        # 角度预测
        angles = self.angle_predictor(aggregated)  # [batch, 2]
        
        return angles

def ultra_loss_function(predictions, targets, loss_type='ultra_focal'):
    """超强损失函数，结合多种损失策略"""
    pred_angles = predictions
    target_theta, target_phi = targets
    
    target_theta_last = target_theta[:, -1]
    target_phi_last = target_phi[:, -1]
    
    if loss_type == 'ultra_focal':
        # 计算基础误差
        theta_error = torch.abs(pred_angles[:, 0] - target_theta_last)
        phi_error = torch.abs(pred_angles[:, 1] - target_phi_last)
        
        # 自适应focal权重
        theta_focal = torch.pow(theta_error + 1e-8, 0.25)  # 更温和的权重
        phi_focal = torch.pow(phi_error + 1e-8, 0.25)
        
        # 组合损失：Huber + MSE + Smooth L1
        theta_huber = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=0.1, reduction='none')
        phi_huber = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=0.1, reduction='none')
        
        theta_mse = F.mse_loss(pred_angles[:, 0], target_theta_last, reduction='none')
        phi_mse = F.mse_loss(pred_angles[:, 1], target_phi_last, reduction='none')
        
        theta_smooth = F.smooth_l1_loss(pred_angles[:, 0], target_theta_last, reduction='none')
        phi_smooth = F.smooth_l1_loss(pred_angles[:, 1], target_phi_last, reduction='none')
        
        # 加权组合
        theta_loss = theta_focal * (0.4 * theta_huber + 0.3 * theta_mse + 0.3 * theta_smooth)
        phi_loss = phi_focal * (0.4 * phi_huber + 0.3 * phi_mse + 0.3 * phi_smooth)
        
        theta_loss = theta_loss.mean()
        phi_loss = phi_loss.mean()
        
        # 角度一致性损失
        angle_consistency = F.mse_loss(
            torch.norm(pred_angles, dim=1),
            torch.norm(torch.stack([target_theta_last, target_phi_last], dim=1), dim=1)
        )
        
        total_loss = theta_loss + phi_loss + 0.1 * angle_consistency
        
    else:
        # 回退到标准损失
        theta_loss = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=1.0)
        phi_loss = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=1.0)
        total_loss = theta_loss + phi_loss
        
    # 确保损失值在合理范围内
    total_loss = torch.clamp(total_loss, 0, 50)
    
    return total_loss, theta_loss, phi_loss

def create_enhanced_model():
    """创建增强模型"""
    model = UltraTransformerBeamformingNet(
        d_model=512,    # 增大模型维度
        nhead=16,        # 增加注意力头数
        num_layers=12,   # 增加层数
        dropout=0.1      # 适度dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔧 增强模型创建完成:")
    print(f"   • 总参数数量: {total_params:,}")
    print(f"   • 可训练参数: {trainable_params:,}")
    print(f"   • 模型维度: {model.d_model}")
    print(f"   • 注意力头数: 16")
    print(f"   • Transformer层数: 12")
    
    return model

def enhanced_train_model(model, train_loader, val_loader, num_epochs=200, lr=1e-4, device='cuda', use_multi_gpu=True):
    """增强训练函数，支持多GPU并行训练"""
    
    # 初始化GPU监控器
    gpu_monitor = GPUMonitor()
    gpu_monitor.reset()
    
    # 优化显存使用
    optimize_memory_usage()
    
    # 设置多GPU训练
    if use_multi_gpu and torch.cuda.device_count() > 1:
        model, device = setup_multi_gpu_model(model)
        effective_batch_size = train_loader.batch_size * torch.cuda.device_count()
        print(f"🔥 多GPU训练 - 有效批次大小: {effective_batch_size}")
    else:
        model.to(device)
        effective_batch_size = train_loader.batch_size
    
    # 检查模型是否被DataParallel包装
    is_multi_gpu = isinstance(model, DataParallel)
    
    # 使用更保守的优化器设置（适应复杂模型）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4,  # 适度正则化
        eps=1e-8,
        betas=(0.9, 0.95)  # 更保守的momentum
    )
    
    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # 每20个epoch重启一次
        T_mult=2,  # 每次重启时周期翻倍
        eta_min=lr/100  # 最小学习率
    )
    
    # 混合精度训练（节省显存和加速）
    scaler = torch.cuda.amp.GradScaler() if device != 'cpu' else None
    use_amp = scaler is not None
    
    # 梯度累积参数（根据GPU数量调整）
    accumulation_steps = max(1, 4 // torch.cuda.device_count()) if torch.cuda.is_available() else 2
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    min_improvement = 1e-6
    
    print(f"🚀 开始增强模型训练:")
    print(f"   • 初始学习率: {lr}")
    print(f"   • 梯度累积步数: {accumulation_steps}")
    print(f"   • 早停耐心值: {early_stop_patience}")
    print(f"   • 混合精度训练: {'启用' if use_amp else '禁用'}")
    print(f"   • 多GPU训练: {'启用' if is_multi_gpu else '禁用'}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_theta_loss = 0
        train_phi_loss = 0
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"🎯 训练 Epoch {epoch+1}/{num_epochs}", leave=False)
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 检查输入数据的有效性
                if torch.isnan(H_input).any() or torch.isinf(H_input).any():
                    print(f"⚠️ 输入数据包含NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                # 混合精度前向传播
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(H_input)
                        loss, theta_loss, phi_loss = ultra_loss_function(
                            predictions, (theta_label, phi_label), loss_type='ultra_focal'
                        )
                        loss = loss / accumulation_steps
                else:
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = ultra_loss_function(
                        predictions, (theta_label, phi_label), loss_type='ultra_focal'
                    )
                    loss = loss / accumulation_steps
                
                # 检查损失的有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 损失为NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                # 反向传播（累积梯度）
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 每accumulation_steps步执行一次优化
                if (batch_idx + 1) % accumulation_steps == 0:
                    if use_amp:
                        # 梯度裁剪
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                train_loss_epoch += loss.item() * accumulation_steps
                train_theta_loss += theta_loss.item()
                train_phi_loss += phi_loss.item()
                num_batches += 1
                
                # 更新GPU监控
                gpu_monitor.update()
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                peak_memory = gpu_monitor.get_peak_memory()
                train_progress.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.4f}',
                    'θ': f'{theta_loss.item():.4f}',
                    'φ': f'{phi_loss.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'GPU': f'{peak_memory:.1f}GB'
                })
                
                # 定期清理显存
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ 训练批次 {batch_idx} 出错: {e}")
                continue
        
        # 处理最后的梯度累积
        if num_batches % accumulation_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()  # 更新学习率
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_theta_loss = 0
        val_phi_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="🔍 验证", leave=False)
            for H_input, theta_label, phi_label, H_complex in val_progress:
                try:
                    H_input = H_input.to(device, non_blocking=True)
                    theta_label = theta_label.to(device, non_blocking=True)
                    phi_label = phi_label.to(device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            predictions = model(H_input)
                            loss, theta_loss, phi_loss = ultra_loss_function(
                                predictions, (theta_label, phi_label), loss_type='ultra_focal'
                            )
                    else:
                        predictions = model(H_input)
                        loss, theta_loss, phi_loss = ultra_loss_function(
                            predictions, (theta_label, phi_label), loss_type='ultra_focal'
                        )
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_theta_loss += theta_loss.item()
                        val_phi_loss += phi_loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    continue
        
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
        
        # 动态调整早停策略
        current_early_stop_patience = early_stop_patience
        if val_loss_avg < 0.0001:  # 当验证损失极低时
            current_early_stop_patience = 80  # 更大的patience
        elif val_loss_avg < 0.001:
            current_early_stop_patience = 60
        
        # 打印详细的epoch结果
        current_lr = optimizer.param_groups[0]['lr']
        train_val_gap = val_loss_avg - train_loss_avg
        peak_memory = gpu_monitor.get_peak_memory()
        
        # 使用emoji和颜色突出重要信息
        status_icon = "🔥" if val_loss_avg < best_val_loss else "📊"
        print(f"{status_icon} Epoch {epoch+1:3d}: "
              f"Train={train_loss_avg:.6f} (θ:{train_theta_avg:.6f}, φ:{train_phi_avg:.6f}) | "
              f"Val={val_loss_avg:.6f} (θ:{val_theta_avg:.6f}, φ:{val_phi_avg:.6f}) | "
              f"Gap={train_val_gap:.6f} | LR={current_lr:.2e} | "
              f"GPU={peak_memory:.1f}GB | Patience={patience_counter}/{current_early_stop_patience}")
        
        # 早停和模型保存
        improvement = best_val_loss - val_loss_avg
        if improvement > min_improvement:
            best_val_loss = val_loss_avg
            patience_counter = 0
            
            # 保存模型时需要考虑DataParallel
            if is_multi_gpu:
                torch.save(model.module.state_dict(), 'best_ultra_beamforming_model.pth')
            else:
                torch.save(model.state_dict(), 'best_ultra_beamforming_model.pth')
            print(f"   ✅ 保存最佳模型 (Val Loss: {val_loss_avg:.6f}, 改进: {improvement:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= current_early_stop_patience:
            print(f"🛑 早停触发！在epoch {epoch+1}停止训练 (patience: {patience_counter}/{current_early_stop_patience})")
            break
        
        # 定期显存清理
        if epoch % 10 == 0:
            optimize_memory_usage()
    
    print(f"🏁 训练完成，峰值显存使用: {gpu_monitor.get_peak_memory():.1f}GB")
    return train_losses, val_losses

def train_model(model, train_loader, val_loader, num_epochs=150, lr=3e-4, device='cuda', use_multi_gpu=True):
    """原始训练函数，支持多GPU并行训练"""
    
    # 初始化GPU监控器
    gpu_monitor = GPUMonitor()
    gpu_monitor.reset()
    
    # 优化显存使用
    optimize_memory_usage()
    
    # 设置多GPU训练
    if use_multi_gpu and torch.cuda.device_count() > 1:
        model, device = setup_multi_gpu_model(model)
        effective_batch_size = train_loader.batch_size * torch.cuda.device_count()
        print(f"🔥 多GPU训练 - 有效批次大小: {effective_batch_size}")
    else:
        model.to(device)
        effective_batch_size = train_loader.batch_size
    
    # 检查模型是否被DataParallel包装
    is_multi_gpu = isinstance(model, DataParallel)
    
    # 使用更激进的优化器设置
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=2e-4,  # 增加正则化防止过拟合
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # 改进的学习率调度策略
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10%时间warm-up
        div_factor=10,  # 初始lr = max_lr/10
        final_div_factor=100,  # 最终lr = max_lr/100
        anneal_strategy='cos'
    )
    
    # 混合精度训练（节省显存和加速）
    scaler = torch.cuda.amp.GradScaler() if device != 'cpu' else None
    use_amp = scaler is not None
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 40  # 增加patience，在低损失时更宽容
    min_improvement = 1e-5  # 最小改进阈值，避免微小波动触发早停
    
    print(f"开始训练，最大学习率: {lr}")
    print(f"   • 混合精度训练: {'启用' if use_amp else '禁用'}")
    print(f"   • 多GPU训练: {'启用' if is_multi_gpu else '禁用'}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_theta_loss = 0
        train_phi_loss = 0
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 检查输入数据的有效性
                if torch.isnan(H_input).any() or torch.isinf(H_input).any():
                    print(f"警告: 输入数据包含NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                optimizer.zero_grad()
                
                # 混合精度前向传播
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(H_input)
                        loss, theta_loss, phi_loss = improved_loss_function(
                            predictions, (theta_label, phi_label), loss_type='focal_huber'
                        )
                else:
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = improved_loss_function(
                        predictions, (theta_label, phi_label), loss_type='focal_huber'
                    )
                
                # 检查损失的有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 损失为NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                # 反向传播
                if use_amp:
                    scaler.scale(loss).backward()
                    # 梯度裁剪（适应更高学习率）
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪（适应更高学习率）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                
                scheduler.step()  # OneCycleLR需要每个step调用
                
                train_loss_epoch += loss.item()
                train_theta_loss += theta_loss.item()
                train_phi_loss += phi_loss.item()
                num_batches += 1
                
                # 更新GPU监控
                gpu_monitor.update()
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                peak_memory = gpu_monitor.get_peak_memory()
                train_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'θ': f'{theta_loss.item():.4f}',
                    'φ': f'{phi_loss.item():.4f}',
                    'LR': f'{current_lr:.2e}',
                    'GPU': f'{peak_memory:.1f}GB'
                })
                
                # 定期清理显存
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_theta_loss = 0
        val_phi_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="验证", leave=False)
            for H_input, theta_label, phi_label, H_complex in val_progress:
                try:
                    H_input = H_input.to(device, non_blocking=True)
                    theta_label = theta_label.to(device, non_blocking=True)
                    phi_label = phi_label.to(device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            predictions = model(H_input)
                            loss, theta_loss, phi_loss = improved_loss_function(
                                predictions, (theta_label, phi_label), loss_type='focal_huber'
                            )
                    else:
                        predictions = model(H_input)
                        loss, theta_loss, phi_loss = improved_loss_function(
                            predictions, (theta_label, phi_label), loss_type='focal_huber'
                        )
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_theta_loss += theta_loss.item()
                        val_phi_loss += phi_loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    continue
        
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
        
        # 在极低损失时使用更宽松的早停策略
        current_early_stop_patience = early_stop_patience
        if val_loss_avg < 0.001:  # 当验证损失很低时
            current_early_stop_patience = 60  # 更大的patience
        
        # 打印epoch结果 - 增加更多监控信息
        current_lr = optimizer.param_groups[0]['lr']
        train_val_gap = val_loss_avg - train_loss_avg
        peak_memory = gpu_monitor.get_peak_memory()
        print(f"Epoch {epoch+1:2d}: Train={train_loss_avg:.6f} (θ:{train_theta_avg:.6f}, φ:{train_phi_avg:.6f}), "
              f"Val={val_loss_avg:.6f} (θ:{val_theta_avg:.6f}, φ:{val_phi_avg:.6f}), "
              f"Gap={train_val_gap:.6f}, LR={current_lr:.2e}, GPU={peak_memory:.1f}GB, Patience={patience_counter}/{current_early_stop_patience}")
        
        # 早停和模型保存 - 改进策略
        improvement = best_val_loss - val_loss_avg
        if improvement > min_improvement:  # 只有明显改进才重置计数器
            best_val_loss = val_loss_avg
            patience_counter = 0
            
            # 保存模型时需要考虑DataParallel
            if is_multi_gpu:
                torch.save(model.module.state_dict(), 'best_beamforming_model_fixed.pth')
            else:
                torch.save(model.state_dict(), 'best_beamforming_model_fixed.pth')
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss_avg:.6f}, 改进: {improvement:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= current_early_stop_patience:
            print(f"早停触发！在epoch {epoch+1}停止训练 (patience: {patience_counter}/{current_early_stop_patience})")
            break
        
        # 定期显存清理
        if epoch % 10 == 0:
            optimize_memory_usage()
    
    print(f" 训练完成，峰值显存使用: {gpu_monitor.get_peak_memory():.1f}GB")
    return train_losses, val_losses

def improved_loss_function(predictions, targets, loss_type='focal_huber'):
    """改进的损失函数，降低角度预测误差"""
    pred_angles = predictions  # [batch, 2] (theta, phi)
    target_theta, target_phi = targets
    
    # 使用最后一个时间步的目标
    target_theta_last = target_theta[:, -1]  # [batch]
    target_phi_last = target_phi[:, -1]      # [batch]
    
    if loss_type == 'focal_huber':
        # 组合Focal Loss和Huber Loss的思想
        # 计算基础误差
        theta_error = torch.abs(pred_angles[:, 0] - target_theta_last)
        phi_error = torch.abs(pred_angles[:, 1] - target_phi_last)
        
        # Focal权重：对难样本加权
        theta_focal_weight = torch.pow(theta_error + 1e-8, 0.5)  # 平方根权重
        phi_focal_weight = torch.pow(phi_error + 1e-8, 0.5)
        
        # Huber损失
        theta_loss = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=0.5, reduction='none')
        phi_loss = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=0.5, reduction='none')
        
        # 应用focal权重
        theta_loss = (theta_focal_weight * theta_loss).mean()
        phi_loss = (phi_focal_weight * phi_loss).mean()
        
    elif loss_type == 'adaptive_mse':
        # 自适应MSE损失
        theta_loss = F.mse_loss(pred_angles[:, 0], target_theta_last)
        phi_loss = F.mse_loss(pred_angles[:, 1], target_phi_last)
        
        # 动态权重平衡
        theta_weight = 1.0 / (theta_loss.detach() + 1e-8)
        phi_weight = 1.0 / (phi_loss.detach() + 1e-8)
        
        total_weight = theta_weight + phi_weight
        theta_weight = theta_weight / total_weight
        phi_weight = phi_weight / total_weight
        
        theta_loss = theta_weight * theta_loss
        phi_loss = phi_weight * phi_loss
        
    else:
        # 标准Huber损失
        theta_loss = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=1.0)
        phi_loss = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=1.0)
    
    total_loss = theta_loss + phi_loss
    
    # 添加正则化项
    l2_reg = 0.0
    for param in pred_angles.requires_grad_(True):
        if param.requires_grad:
            l2_reg += torch.norm(param)
    total_loss += 1e-6 * l2_reg
    
    # 确保损失值在合理范围内
    total_loss = torch.clamp(total_loss, 0, 50)
    
    return total_loss, theta_loss, phi_loss

def create_original_model():
    """创建原始模型（用于对比）"""
    model = TransformerBeamformingNet(
        input_dim=256*256*2,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.15
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔧 原始模型创建完成:")
    print(f"   • 总参数数量: {total_params:,}")
    print(f"   • 模型维度: 512")
    print(f"   • 注意力头数: 8")
    print(f"   • Transformer层数: 6")
    
    return model

def test_model(model, test_loader, device='cuda'):
    """测试模型性能，支持多GPU"""
    
    # 检查模型是否被DataParallel包装
    is_multi_gpu = isinstance(model, DataParallel)
    
    # 如果是多GPU模型，确保设备正确
    if is_multi_gpu:
        model.to(device)
    else:
        model.to(device)
    
    model.eval()
    
    all_predictions = []
    angle_errors = []
    
    # 启用混合精度推理以节省显存
    use_amp = torch.cuda.is_available() and device != 'cpu'
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="🧪 测试进度")
        
        for H_input, theta_label, phi_label, H_complex in test_progress:
            try:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                # 混合精度推理
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(H_input)
                else:
                    predictions = model(H_input)
                
                # 转换为度数（反归一化）
                pred_angles = predictions.cpu().numpy()
                pred_theta_norm = pred_angles[:, 0]  # [-1, 1]
                pred_phi_norm = pred_angles[:, 1]    # [-1, 1]
                
                # 反归一化到度数（修正范围）
                pred_theta_deg = pred_theta_norm * 90  # [-90, 90] 对应 [-1, 1]
                pred_phi_deg = pred_phi_norm * 180     # [-180, 180] 对应 [-1, 1]
                
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
        
        print(f"\n📊 测试结果:")
        print(f"平均角度误差: {mean_error:.2f}°")
        print(f"角度误差标准差: {std_error:.2f}°")
        print(f"角度误差中位数: {median_error:.2f}°")
        print(f"有效预测数量: {len(all_predictions)}")
        
        if len(all_predictions) > 0:
            theta_errors = [p['theta_error'] for p in all_predictions]
            phi_errors = [p['phi_error'] for p in all_predictions]
            print(f"θ误差: {np.mean(theta_errors):.2f}° ± {np.std(theta_errors):.2f}°")
            print(f"φ误差: {np.mean(phi_errors):.2f}° ± {np.std(phi_errors):.2f}°")
    else:
        print("没有有效的测试结果")
    
    return all_predictions

def setup_distributed_training():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return None, None, None

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_available_gpus():
    """获取可用的GPU信息"""
    if not torch.cuda.is_available():
        return []
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        gpu_name = torch.cuda.get_device_properties(i).name
        gpu_info.append({
            'id': i,
            'name': gpu_name,
            'memory_gb': gpu_memory
        })
    
    return gpu_info

def setup_multi_gpu_model(model, device_ids=None):
    """设置多GPU模型"""
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，使用CPU训练")
        return model, 'cpu'
    
    gpu_info = get_available_gpus()
    gpu_count = len(gpu_info)
    
    print(f"🖥️  检测到 {gpu_count} 张GPU:")
    for gpu in gpu_info:
        print(f"   • GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    
    if gpu_count >= 2:
        if device_ids is None:
            device_ids = list(range(gpu_count))
        
        print(f"🚀 启用多GPU并行训练，使用GPU: {device_ids}")
        
        # 将模型移到主GPU
        model = model.cuda(device_ids[0])
        
        # 使用DataParallel进行多GPU训练
        model = DataParallel(model, device_ids=device_ids)
        
        # 设置主设备
        device = f'cuda:{device_ids[0]}'
        
        # 启用混合精度训练以节省显存
        torch.backends.cudnn.benchmark = True
        
        return model, device
    
    elif gpu_count == 1:
        print("📱 使用单GPU训练")
        device = 'cuda:0'
        model = model.cuda()
        return model, device
    
    else:
        print("❌ 没有可用的GPU，使用CPU训练")
        return model, 'cpu'

def optimize_memory_usage():
    """优化显存使用"""
    if torch.cuda.is_available():
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 设置内存分配策略
        torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        
        # 打印当前显存使用情况
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i} 显存: {allocated:.1f}GB 已分配, {cached:.1f}GB 已缓存, {total:.1f}GB 总计")

class GPUMonitor:
    """GPU显存监控器"""
    def __init__(self):
        self.peak_memory = 0
        
    def update(self):
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
    
    def get_peak_memory(self):
        return self.peak_memory
    
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0

def main():
    """主函数 - 支持模型选择和多GPU训练"""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 检测并显示GPU信息
    gpu_info = get_available_gpus()
    gpu_count = len(gpu_info)
    
    print(f"🖥️  GPU检测结果:")
    if gpu_count == 0:
        print("   ❌ 未检测到CUDA GPU，将使用CPU训练")
        device = 'cpu'
        use_multi_gpu = False
    else:
        print(f"   ✅ 检测到 {gpu_count} 张GPU:")
        total_memory = 0
        for gpu in gpu_info:
            print(f"      • GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
            total_memory += gpu['memory_gb']
        print(f"   📊 总显存容量: {total_memory:.1f}GB")
        
        device = 'cuda'
        use_multi_gpu = gpu_count > 1
        
        if use_multi_gpu:
            print(f"   🚀 将启用多GPU并行训练，使用 {gpu_count} 张GPU")
        else:
            print(f"   � 将使用单GPU训练")
    
    # 优化显存使用
    if device == 'cuda':
        optimize_memory_usage()
    
    # 用户选择模型类型
    print("\n🤖 请选择要使用的模型:")
    print("1. 原始模型 (较快训练，适中精度)")
    print("2. 增强模型 (较慢训练，更高精度)")
    
    choice = input("请输入选择 (1 或 2，默认为 2): ").strip()
    use_enhanced = choice != "1"
    
    if use_enhanced:
        print("✅ 选择了增强模型 - 将获得更高精度但需要更长训练时间")
        model_save_name = 'best_ultra_beamforming_model.pth'
        results_file_name = 'enhanced_training_results.png'
    else:
        print("✅ 选择了原始模型 - 训练速度较快")
        model_save_name = 'best_beamforming_model_fixed.pth'
        results_file_name = 'original_training_results.png'
    
    # 数据路径
    channel_folder = "samples_data/channel_data_opt_20250702_143327"
    angle_folder = "samples_data/angle_data_opt_20250702_143327"
    
    print("正在加载数据集...")
    
    # 创建数据集
    dataset = BeamformingDataset(channel_folder, angle_folder)
    
    # 按要求划分数据集：800训练，200测试
    total_samples = len(dataset)
    train_size = 800
    test_size = total_samples - train_size
    
    indices = list(range(total_samples))
    np.random.seed(42)  # 设置随机种子确保结果可重现
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 从训练集中划分验证集 - 增加验证集大小提高稳定性
    val_size = int(train_size * 0.25)  # 25%作为验证集，提高稳定性
    train_final_indices = train_indices[val_size:]
    val_indices = train_indices[:val_size]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_final_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本") 
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 根据GPU数量调整批次大小
    if gpu_count >= 2:
        batch_size = 8  # 双GPU可以使用更大批次
        num_workers = 8  # 增加数据加载线程
    elif gpu_count == 1:
        batch_size = 6  # 单GPU适中批次
        num_workers = 4
    else:
        batch_size = 2  # CPU训练使用小批次
        num_workers = 2
    
    print(f"🔧 训练配置:")
    print(f"   • 批次大小: {batch_size}")
    print(f"   • 数据加载进程: {num_workers}")
    print(f"   • 多GPU训练: {'启用' if use_multi_gpu else '禁用'}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=(device=='cuda'), 
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=(device=='cuda'),
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=(device=='cuda'),
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 创建模型
    print(f"\n🏗️  创建{'增强' if use_enhanced else '原始'}模型...")
    if use_enhanced:
        model = create_enhanced_model()
        
        # 训练模型
        print("\n🚀 开始增强模型训练...")
        train_losses, val_losses = enhanced_train_model(
            model, train_loader, val_loader, 
            num_epochs=200,  # 增加训练轮数
            lr=1e-4,         # 保守的学习率
            device=device,
            use_multi_gpu=use_multi_gpu
        )
    else:
        model = create_original_model()
        
        # 训练模型
        print("\n🚀 开始原始模型训练...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=150,
            lr=3e-4,
            device=device,
            use_multi_gpu=use_multi_gpu
        )
    
    # 加载最佳模型
    if os.path.exists(model_save_name):
        # 处理多GPU模型的状态字典加载
        state_dict = torch.load(model_save_name, map_location=device)
        
        # 如果当前模型是DataParallel但保存的不是，需要添加module前缀
        if isinstance(model, DataParallel) and not any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        # 如果当前模型不是DataParallel但保存的是，需要移除module前缀
        elif not isinstance(model, DataParallel) and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print(f"✅ 加载最佳{'增强' if use_enhanced else '原始'}模型完成")
    
    # 测试模型
    print("\n🧪 开始测试...")
    predictions = test_model(model, test_loader, device=device)
    
    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 角度预测散点图
    if len(predictions) > 0:
        pred_theta = [p['predicted_theta'] for p in predictions]
        target_theta = [p['target_theta'] for p in predictions]
        pred_phi = [p['predicted_phi'] for p in predictions]
        target_phi = [p['target_phi'] for p in predictions]
        
        # θ角度预测
        axes[0, 1].scatter(target_theta, pred_theta, alpha=0.6, s=20)
        min_theta = min(min(target_theta), min(pred_theta))
        max_theta = max(max(target_theta), max(pred_theta))
        axes[0, 1].plot([min_theta, max_theta], [min_theta, max_theta], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('True θ (degrees)')
        axes[0, 1].set_ylabel('Predicted θ (degrees)')
        axes[0, 1].set_title('θ Angle Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # φ角度预测
        axes[1, 0].scatter(target_phi, pred_phi, alpha=0.6, s=20)
        min_phi = min(min(target_phi), min(pred_phi))
        max_phi = max(max(target_phi), max(pred_phi))
        axes[1, 0].plot([min_phi, max_phi], [min_phi, max_phi], 'r--', label='Perfect')
        axes[1, 0].set_xlabel('True φ (degrees)')
        axes[1, 0].set_ylabel('Predicted φ (degrees)')
        axes[1, 0].set_title('φ Angle Prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差分布
        errors = [p['total_error'] for p in predictions]
        axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}°')
        axes[1, 1].set_xlabel('Total Angle Error (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_file_name, dpi=300, bbox_inches='tight')
    print(f"\n📈 {'增强' if use_enhanced else '原始'}模型训练结果图表已保存为 '{results_file_name}'")
    
    # 打印训练总结
    print(f"\n📋 {'增强' if use_enhanced else '原始'}模型训练总结:")
    if train_losses:
        print(f"• 最终训练损失: {train_losses[-1]:.6f}")
        print(f"• 最终验证损失: {val_losses[-1]:.6f}")
        print(f"• 总训练轮数: {len(train_losses)}")
        print(f"• 损失改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
        
        # 计算模型参数数量时要考虑DataParallel
        if isinstance(model, DataParallel):
            param_count = sum(p.numel() for p in model.module.parameters())
        else:
            param_count = sum(p.numel() for p in model.parameters())
        print(f"• 模型复杂度: ~{param_count:,} 参数")
        print(f"• 多GPU训练: {'是' if use_multi_gpu else '否'}")
        print(f"• GPU数量: {gpu_count}")
    
    # 最终显存清理
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"• 最终显存清理完成")
    
    return model, predictions

if __name__ == "__main__":
    main()