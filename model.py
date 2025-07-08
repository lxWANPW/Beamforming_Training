import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class AntennaArrayDataset(Dataset):
    """天线阵列数据集类"""
    def __init__(self, channel_folder, angle_folder, sequence_length=100):
        self.channel_folder = channel_folder
        self.sequence_length = sequence_length
        
        # 加载角度数据
        angle_data = sio.loadmat(os.path.join(angle_folder, 'angle_data.mat'))
        self.theta_array = angle_data['theta_array']
        self.phi_array = angle_data['phi_array']
        self.num_samples = angle_data['num_samples'][0, 0]
        
        # 存储文件路径
        self.file_paths = []
        for i in range(1, self.num_samples + 1):
            filename = f'H_sample_{i:02d}.mat'
            self.file_paths.append(os.path.join(channel_folder, filename))
    
    def __len__(self):
        return int(self.num_samples) * (3000 // self.sequence_length)
    
    def __getitem__(self, idx):
        # 计算样本索引和时间段索引
        sample_idx = idx // (3000 // self.sequence_length)
        time_idx = idx % (3000 // self.sequence_length)
        
        # 加载信道数据
        channel_data = sio.loadmat(self.file_paths[sample_idx])
        H_real = channel_data['H_real']
        H_imag = channel_data['H_imag']
        
        # 提取时间段
        start_idx = time_idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        
        # 组合实部和虚部
        H_complex = H_real[:, :, start_idx:end_idx] + 1j * H_imag[:, :, start_idx:end_idx]
        
        # 转换为实数表示 [2, 256, 256, seq_len]
        H_tensor = np.stack([H_real[:, :, start_idx:end_idx], 
                            H_imag[:, :, start_idx:end_idx]], axis=0)
        
        # 获取对应角度
        theta = self.theta_array[sample_idx, start_idx:end_idx]
        phi = self.phi_array[sample_idx, start_idx:end_idx]
        
        return {
            'channel': torch.FloatTensor(H_tensor),
            'theta': torch.FloatTensor(theta),
            'phi': torch.FloatTensor(phi),
            'H_complex': H_complex
        }

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SubarrayPartitionModule(nn.Module):
    """子阵列划分模块"""
    def __init__(self, num_antennas=256, hidden_dim=512):
        super().__init__()
        self.num_antennas = num_antennas
        
        # 学习子阵列划分权重
        self.partition_net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_antennas),
            nn.Sigmoid()
        )
        
        # 预定义多种划分模式
        self.register_buffer('partition_modes', self._create_partition_modes())
    
    def _create_partition_modes(self):
        """创建预定义的子阵列划分模式"""
        # 只返回4x4子阵列模式 (16个子阵列)
        mode = torch.zeros(256, 16)
        for i in range(16):
            row = (i // 4) * 4
            col = (i % 4) * 4
            for r in range(4):
                for c in range(4):
                    antenna_idx = (row + r) * 16 + (col + c)
                    mode[antenna_idx, i] = 1
        
        return mode
    
    def forward(self, features):
        partition_weights = self.partition_net(features)
        
        # 使用预定义的4x4划分模式
        best_partition = self.partition_modes
        
        return partition_weights, best_partition

class BeamformingTransformer(nn.Module):
    """多任务波束成形Transformer模型"""
    def __init__(self, num_antennas=256, d_model=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()
        
        self.num_antennas = num_antennas
        self.d_model = d_model
        
        # 输入编码器
        self.channel_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, d_model)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        
        # 任务特定的解码头
        # 1. 角度预测头
        self.angle_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # theta, phi
        )
        
        # 2. 波束成形参数头
        self.beamforming_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, num_antennas * 2)  # 复数权重
        )
        
        # 3. 子阵列划分模块
        self.subarray_module = SubarrayPartitionModule(num_antennas, d_model)
        
        # 4. 波束特性预测头
        self.beam_property_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 主瓣宽度、旁瓣抑制、SNR增益
        )
    
    def forward(self, x):
        batch_size, _, _, _, seq_len = x.shape
        
        # 编码每个时间步的信道
        encoded_seq = []
        for t in range(seq_len):
            channel_t = x[:, :, :, :, t]
            encoded_t = self.channel_encoder(channel_t)
            encoded_seq.append(encoded_t)
        
        # 堆叠时间序列 [seq_len, batch, d_model]
        src = torch.stack(encoded_seq, dim=0)
        
        # 添加位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        memory = self.transformer_encoder(src)
        
        # 聚合时间信息
        aggregated = torch.mean(memory, dim=0)  # [batch, d_model]
        
        # 多任务输出
        # 1. 角度预测
        angles = self.angle_predictor(aggregated)
        theta_pred = angles[:, 0] * 90  # 归一化到[0, 90]
        phi_pred = angles[:, 1] * 360   # 归一化到[0, 360]
        
        # 2. 波束成形权重
        beamforming_weights = self.beamforming_head(aggregated)
        weights_complex = beamforming_weights[:, :self.num_antennas] + \
                         1j * beamforming_weights[:, self.num_antennas:]
        weights_complex = weights_complex / (torch.abs(weights_complex) + 1e-8)
        
        # 3. 子阵列划分
        partition_weights, best_partition = self.subarray_module(aggregated)
        
        # 4. 波束特性
        beam_properties = self.beam_property_head(aggregated)
        
        return {
            'theta': theta_pred,
            'phi': phi_pred,
            'beamforming_weights': weights_complex,
            'partition_weights': partition_weights,
            'best_partition': best_partition,
            'beam_properties': beam_properties
        }

def compute_beam_pattern(weights, theta_range, phi_target, num_ant_x=16, num_ant_y=16, d=0.5):
    """计算波束图"""
    pattern = []
    for theta in theta_range:
        az = phi_target * np.pi / 180
        el = theta * np.pi / 180
        
        # 天线阵列响应
        k = 2 * np.pi * np.array([np.cos(el)*np.cos(az), 
                                  np.cos(el)*np.sin(az), 
                                  np.sin(el)])
        
        [x_idx, y_idx] = np.meshgrid(range(num_ant_x), range(num_ant_y))
        ant_pos = np.column_stack([x_idx.flatten(), y_idx.flatten()]) * d
        
        response = np.exp(1j * (ant_pos @ k[:2]))
        
        # 应用权重
        beam_response = np.abs(np.sum(weights * response))**2
        pattern.append(beam_response)
    
    pattern = np.array(pattern)
    pattern_dB = 10 * np.log10(pattern / np.max(pattern) + 1e-10)
    
    return pattern_dB

def calculate_beam_metrics(pattern_dB):
    """计算波束指标"""
    # 主瓣宽度 (3dB)
    main_lobe_idx = np.argmax(pattern_dB)
    half_power = -3
    indices = np.where(pattern_dB >= half_power)[0]
    if len(indices) > 0:
        beamwidth = indices[-1] - indices[0]
    else:
        beamwidth = 1
    
    # 旁瓣抑制
    # 找到第一个零点
    zero_crossings = np.where(np.diff(np.sign(pattern_dB[main_lobe_idx:] + 20)))[0]
    if len(zero_crossings) > 0:
        first_null = main_lobe_idx + zero_crossings[0]
        if first_null < len(pattern_dB) - 1:
            sidelobe_level = np.max(pattern_dB[first_null:])
        else:
            sidelobe_level = -30
    else:
        sidelobe_level = -20
    
    return beamwidth, abs(sidelobe_level)

class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    def __init__(self):
        super().__init__()
        self.angle_loss_weight = 1.0
        self.beam_property_weight = 0.5
        self.snr_weight = 0.3
        self.partition_weight = 0.2
    
    def forward(self, outputs, targets, H_complex):
        # 1. 角度预测损失
        angle_loss = F.mse_loss(outputs['theta'], targets['theta']) + \
                     F.mse_loss(outputs['phi'], targets['phi'])
        
        # 2. 波束特性损失（需要计算实际波束图）
        batch_size = outputs['theta'].shape[0]
        beam_metrics_loss = 0
        
        for i in range(batch_size):
            weights = outputs['beamforming_weights'][i].detach().cpu().numpy()
            theta_target = targets['theta'][i].item()
            phi_target = targets['phi'][i].item()
            
            # 计算波束图
            theta_range = np.linspace(0, 90, 181)
            pattern_dB = compute_beam_pattern(weights, theta_range, phi_target)
            
            # 计算指标
            beamwidth, sidelobe_suppression = calculate_beam_metrics(pattern_dB)
            
            # 期望的指标
            target_beamwidth = 10  # 度
            target_sidelobe = 30   # dB
            
            beam_metrics_loss += (beamwidth - target_beamwidth)**2 / 100 + \
                                (sidelobe_suppression - target_sidelobe)**2 / 100
        
        beam_metrics_loss /= batch_size
        
        # 3. SNR最大化损失（负SNR）
        snr_loss = -torch.mean(outputs['beam_properties'][:, 2])
        
        # 4. 子阵列划分正则化
        partition_loss = -torch.mean(outputs['partition_weights'] * 
                                   torch.log(outputs['partition_weights'] + 1e-8))
        
        # 总损失
        total_loss = self.angle_loss_weight * angle_loss + \
                    self.beam_property_weight * beam_metrics_loss + \
                    self.snr_weight * snr_loss + \
                    self.partition_weight * partition_loss
        
        return total_loss, {
            'angle_loss': angle_loss.item(),
            'beam_metrics_loss': beam_metrics_loss,
            'snr_loss': snr_loss.item(),
            'partition_loss': partition_loss.item()
        }

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """训练函数"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = MultiTaskLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_metrics = {'angle_loss': 0, 'beam_metrics_loss': 0, 
                        'snr_loss': 0, 'partition_loss': 0}
        
        for batch in train_loader:
            channel = batch['channel'].to(device)
            theta = batch['theta'].mean(dim=1).to(device)
            phi = batch['phi'].mean(dim=1).to(device)
            H_complex = batch['H_complex']
            
            optimizer.zero_grad()
            outputs = model(channel)
            
            targets = {'theta': theta, 'phi': phi}
            loss, metrics = criterion(outputs, targets, H_complex)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics[k] += v
        
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_metrics = {'angle_loss': 0, 'beam_metrics_loss': 0, 
                      'snr_loss': 0, 'partition_loss': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                channel = batch['channel'].to(device)
                theta = batch['theta'].mean(dim=1).to(device)
                phi = batch['phi'].mean(dim=1).to(device)
                H_complex = batch['H_complex']
                
                outputs = model(channel)
                targets = {'theta': theta, 'phi': phi}
                loss, metrics = criterion(outputs, targets, H_complex)
                
                val_loss += loss.item()
                for k, v in metrics.items():
                    val_metrics[k] += v
        
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Val Metrics: {val_metrics}")
        print("-" * 50)
    
    return train_losses, val_losses

def test_model(model, test_loader, device='cuda'):
    """测试函数"""
    model = model.to(device)
    model.eval()
    
    all_theta_true = []
    all_theta_pred = []
    all_phi_true = []
    all_phi_pred = []
    all_beamwidths = []
    all_sidelobes = []
    all_snr_gains = []
    
    with torch.no_grad():
        for batch in test_loader:
            channel = batch['channel'].to(device)
            theta_true = batch['theta'].mean(dim=1).cpu().numpy()
            phi_true = batch['phi'].mean(dim=1).cpu().numpy()
            
            outputs = model(channel)
            
            theta_pred = outputs['theta'].cpu().numpy()
            phi_pred = outputs['phi'].cpu().numpy()
            beam_properties = outputs['beam_properties'].cpu().numpy()
            
            all_theta_true.extend(theta_true)
            all_theta_pred.extend(theta_pred)
            all_phi_true.extend(phi_true)
            all_phi_pred.extend(phi_pred)
            
            # 计算实际波束指标
            for i in range(len(theta_pred)):
                weights = outputs['beamforming_weights'][i].cpu().numpy()
                theta_range = np.linspace(0, 90, 181)
                pattern_dB = compute_beam_pattern(weights, theta_range, phi_pred[i])
                beamwidth, sidelobe = calculate_beam_metrics(pattern_dB)
                
                all_beamwidths.append(beamwidth)
                all_sidelobes.append(sidelobe)
                all_snr_gains.append(beam_properties[i, 2])
    
    # 计算统计指标
    theta_mae = np.mean(np.abs(np.array(all_theta_true) - np.array(all_theta_pred)))
    phi_mae = np.mean(np.abs(np.array(all_phi_true) - np.array(all_phi_pred)))
    avg_beamwidth = np.mean(all_beamwidths)
    avg_sidelobe = np.mean(all_sidelobes)
    avg_snr_gain = np.mean(all_snr_gains)
    
    print(f"测试结果：")
    print(f"Theta MAE: {theta_mae:.2f}°")
    print(f"Phi MAE: {phi_mae:.2f}°")
    print(f"平均主瓣宽度: {avg_beamwidth:.2f}°")
    print(f"平均旁瓣抑制: {avg_sidelobe:.2f} dB")
    print(f"平均SNR增益: {avg_snr_gain:.2f} dB")
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 角度预测散点图
    axes[0, 0].scatter(all_theta_true, all_theta_pred, alpha=0.5)
    axes[0, 0].plot([0, 90], [0, 90], 'r--')
    axes[0, 0].set_xlabel('True Theta (°)')
    axes[0, 0].set_ylabel('Predicted Theta (°)')
    axes[0, 0].set_title('Theta Prediction')
    
    axes[0, 1].scatter(all_phi_true, all_phi_pred, alpha=0.5)
    axes[0, 1].plot([0, 360], [0, 360], 'r--')
    axes[0, 1].set_xlabel('True Phi (°)')
    axes[0, 1].set_ylabel('Predicted Phi (°)')
    axes[0, 1].set_title('Phi Prediction')
    
    # 波束特性直方图
    axes[1, 0].hist(all_beamwidths, bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Beamwidth (°)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Beamwidth Distribution')
    
    axes[1, 1].hist(all_sidelobes, bins=30, alpha=0.7)
    axes[1, 1].set_xlabel('Sidelobe Suppression (dB)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Sidelobe Suppression Distribution')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()
    
    return {
        'theta_mae': theta_mae,
        'phi_mae': phi_mae,
        'avg_beamwidth': avg_beamwidth,
        'avg_sidelobe': avg_sidelobe,
        'avg_snr_gain': avg_snr_gain
    }

# 使用示例
if __name__ == "__main__":
    # 设置参数
    channel_folder = "D:/大论文相关/Beamforming-Training/samples_data/channel_data_20250618_145324"  # 替换为实际文件夹名
    angle_folder = "D:/大论文相关/Beamforming-Training/samples_data/angle_data_20250618_145324"      # 替换为实际文件夹名
    
    # 创建数据集
    dataset = AntennaArrayDataset(channel_folder, angle_folder, sequence_length=100)
    
    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 创建模型
    model = BeamformingTransformer(
        num_antennas=256,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048
    )
    
    # 训练模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                          num_epochs=50, device=device)
    
    # 测试模型
    test_results = test_model(model, test_loader, device=device)
    
    # 保存模型
    torch.save(model.state_dict(), 'beamforming_transformer_model.pth')
    
    # 获取最优子阵列划分
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        channel = sample_batch['channel'].to(device)
        outputs = model(channel)
        best_partition = outputs['best_partition'][0].cpu().numpy()
        
        print("\n最优子阵列划分方法：")
        print(f"划分矩阵形状: {best_partition.shape}")
        print(f"子阵列数量: {best_partition.shape[1]}")