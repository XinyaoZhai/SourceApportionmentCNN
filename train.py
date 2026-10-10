import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import pickle
import gc
import os
import pandas as pd

print("已完成导入库")

# ==================== 变量 ====================
TARGET_VAR = 'PN4_G1'
TARGET_VAR_LOWER = 'PN4'
PSAT_VAR = 'PN4_G1'
SOURCE_NAME = 'FERTILIZER' 
EMISSION_VAR = 'NH3'

# 气象变量列表
MET_VARIABLES = ['PBLH', 'PSFC', 'RAIN', 'RH', 'T2', 'WD', 'WS']

# 输入数据通道数
INPUT_CHANNELS = 9  

# 输出目标尺寸
TARGET_HEIGHT = 50
TARGET_WIDTH = 50

# 数据路径
BASE_DATA_DIR = "./data_50"
CAMX_DATA_DIR = f"{BASE_DATA_DIR}/CAMX"
MEIC_DATA_DIR = f"{BASE_DATA_DIR}/MEIC"
WRF_DATA_DIR = f"{BASE_DATA_DIR}/WRF"

# ==================== 设备设置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== 输出目录 ====================
output_dir = Path("./train_output")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

class CAMxDataset(Dataset):
    def __init__(self, input_data, target_data):
        """
        数据集类 - 优化内存使用
        """
        # 使用numpy数组而不是直接转换为tensor，减少内存占用
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        # 在获取时转换为tensor
        input_tensor = torch.FloatTensor(self.input_data[idx])
        target_tensor = torch.FloatTensor(self.target_data[idx])
        return input_tensor, target_tensor

class SourceApportionmentCNN(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=1, 
                 target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):
        super(SourceApportionmentCNN, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        
        # 编码器部分 - 提取特征
        self.encoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 解码器部分 - 上采样恢复分辨率
        self.decoder = nn.Sequential(
            # 第一次上采样 - 使用转置卷积
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第二次上采样 - 使用转置卷积
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 最终调整层确保输出尺寸正确
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.ReLU()  # 使用ReLU确保输出非负
        )
        
        # 如果尺寸仍然不匹配，添加最终调整层
        self.final_adjust = nn.Sequential(
            nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # 确保输出尺寸与目标一致
        if x.shape[2] != self.target_height or x.shape[3] != self.target_width:
            x = self.final_adjust(x)
        return x

def load_and_preprocess_data():
    """加载和预处理所有数据，并保留原始数据维度信息"""
    print("Loading data...")
    
    # 构建文件路径
    camx_psat_path = f"{CAMX_DATA_DIR}/CN03km_{PSAT_VAR}_rowcol_202401.nc"
    meic_path = f"{MEIC_DATA_DIR}/MEIC202401_{EMISSION_VAR}_{SOURCE_NAME}_rowcol.nc"
    camx_target_path = f"{CAMX_DATA_DIR}/CN03km_{TARGET_VAR}_22sector_rowcol_202401.nc"
    
    print(f"Loading PSAT data from: {camx_psat_path}")
    print(f"Loading {SOURCE_NAME} emission data from: {meic_path}")
    print(f"Loading target data from: {camx_target_path}")
    
    # 加载目标数据，同时获取坐标信息
    with xr.open_dataset(camx_target_path) as ds:
        target_data = ds[TARGET_VAR].values  
        time_coords = ds.time.values
        window_start_row = ds.window_start_row.values
        window_start_col = ds.window_start_col.values
        print(f"Target data shape: {target_data.shape}")
    
    # 加载输入数据
    input_datasets = []
    
    # 1. PSAT源解析
    with xr.open_dataset(camx_psat_path) as ds:
        psat_data = ds[PSAT_VAR].values
        input_datasets.append(psat_data)
        print(f"PSAT data shape: {psat_data.shape}")
    
    # 2. 小源排放 
    with xr.open_dataset(meic_path) as ds:
        emission = ds[EMISSION_VAR].values
        input_datasets.append(emission)
        print(f"{EMISSION_VAR} {SOURCE_NAME} emission shape: {emission.shape}")
    
    # 3. 气象参数
    for var in MET_VARIABLES:
        wrf_path = f"{WRF_DATA_DIR}/CN03_WRF_{var}_202401.nc"
        with xr.open_dataset(wrf_path) as ds:
            met_data = ds[var].values
            input_datasets.append(met_data)
            print(f"{var} shape: {met_data.shape}")
    
    # 合并所有输入数据
    # 将所有数据重塑为 [time * window_row * window_col, channels, win_row, win_col]
    num_samples = target_data.shape[0] * target_data.shape[1] * target_data.shape[2]
    num_channels = len(input_datasets)
    height, width = target_data.shape[3], target_data.shape[4]
    
    input_combined = np.zeros((num_samples, num_channels, height, width))
    target_combined = np.zeros((num_samples, 1, height, width))
    
    # 记录每个样本的原始维度索引
    sample_info = []
    
    sample_idx = 0
    for t in range(target_data.shape[0]):
        for wr in range(target_data.shape[1]):
            for wc in range(target_data.shape[2]):
                # 组合所有输入通道
                for ch, data in enumerate(input_datasets):
                    input_combined[sample_idx, ch] = data[t, wr, wc]
                
                # 目标数据
                target_combined[sample_idx, 0] = target_data[t, wr, wc]
                
                # 记录样本信息
                sample_info.append({
                    'time_idx': t,
                    'window_row_idx': wr,
                    'window_col_idx': wc,
                    'time': time_coords[t] if t < len(time_coords) else None
                })
                sample_idx += 1
    
    print(f"Final input shape: {input_combined.shape}")
    print(f"Final target shape: {target_combined.shape}")
    
    # 保存原始维度信息
    original_dims = {
        'time': time_coords,
        'window_start_row': window_start_row,
        'window_start_col': window_start_col,
        'target_shape': target_data.shape,
        'sample_info': sample_info
    }
    
    # 及时清理内存
    del input_datasets
    gc.collect()
    
    return input_combined, target_combined, original_dims

def normalize_data(train_data, val_data=None, test_data=None):
    """标准化数据，仅使用训练集计算参数"""
    normalization_params = {}
    normalized_train = np.zeros_like(train_data)
    
    # 对每个通道进行标准化
    for i in range(train_data.shape[1]):
        channel_data = train_data[:, i]
        
        # 处理NaN值
        channel_data = np.nan_to_num(channel_data, nan=0.0)
        
        # 计算训练集的均值和标准差
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        normalization_params[f'channel_{i}_mean'] = mean_val
        normalization_params[f'channel_{i}_std'] = std_val
        
        # 标准化训练数据
        if std_val > 0:
            normalized_train[:, i] = (channel_data - mean_val) / std_val
        else:
            normalized_train[:, i] = channel_data - mean_val
    
    # 如果有验证数据，使用训练集的参数进行标准化
    normalized_val = None
    if val_data is not None:
        normalized_val = np.zeros_like(val_data)
        for i in range(val_data.shape[1]):
            channel_data = np.nan_to_num(val_data[:, i], nan=0.0)
            mean_val = normalization_params[f'channel_{i}_mean']
            std_val = normalization_params[f'channel_{i}_std']
            
            if std_val > 0:
                normalized_val[:, i] = (channel_data - mean_val) / std_val
            else:
                normalized_val[:, i] = channel_data - mean_val
    
    # 如果有测试数据，使用训练集的参数进行标准化
    normalized_test = None
    if test_data is not None:
        normalized_test = np.zeros_like(test_data)
        for i in range(test_data.shape[1]):
            channel_data = np.nan_to_num(test_data[:, i], nan=0.0)
            mean_val = normalization_params[f'channel_{i}_mean']
            std_val = normalization_params[f'channel_{i}_std']
            
            if std_val > 0:
                normalized_test[:, i] = (channel_data - mean_val) / std_val
            else:
                normalized_test[:, i] = channel_data - mean_val
    
    return normalized_train, normalized_val, normalized_test, normalization_params

def calculate_ioa(predictions, targets):
    """计算一致性指数 (Index of Agreement)"""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return np.nan
    
    mean_obs = np.mean(targets)
    numerator = np.sum((predictions - targets) ** 2)
    denominator = np.sum((np.abs(predictions - mean_obs) + np.abs(targets - mean_obs)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    ioa = 1 - numerator / denominator
    return ioa

def calculate_metrics(predictions, targets, prefix=""):
    """计算评估指标"""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # 移除NaN值
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    # 避免除零错误
    if len(predictions) == 0:
        return {
            f'{prefix}MSE': np.nan, 
            f'{prefix}RMSE': np.nan, 
            f'{prefix}MAE': np.nan, 
            f'{prefix}R2': np.nan, 
            f'{prefix}MAPE': np.nan,
            f'{prefix}IOA': np.nan,
            f'{prefix}MEAN_PRED': np.nan,
            f'{prefix}MEAN_TARGET': np.nan,
            f'{prefix}MB': np.nan
        }
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    ioa = calculate_ioa(predictions, targets)
    
    # 计算平均值
    mean_pred = np.mean(predictions)
    mean_target = np.mean(targets)
    
    # 计算平均偏差 (Mean Bias)
    mb = mean_pred - mean_target
    
    # 计算平均绝对百分比误差，避免除零
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
    
    return {
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}R2': r2,
        f'{prefix}MAPE': mape,
        f'{prefix}IOA': ioa,
        f'{prefix}MEAN_PRED': mean_pred,
        f'{prefix}MEAN_TARGET': mean_target,
        f'{prefix}MB': mb
    }

def plot_loss_curves(train_losses, val_losses, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title(f'Training and Validation Loss ({TARGET_VAR} source)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")

def evaluate_on_test_set(model, test_loader, criterion, device, original_dims, normalization_params):
    """在测试集上评估模型并输出预测结果"""
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc="Testing"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # 检查尺寸是否匹配
            if outputs.shape != batch_y.shape:
                outputs = nn.functional.interpolate(outputs, size=batch_y.shape[2:], mode='bilinear', align_corners=True)
            
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
            # 收集预测和目标用于计算指标
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    test_metrics = calculate_metrics(all_preds, all_targets, prefix="TEST_")
    test_metrics['TEST_LOSS'] = avg_test_loss
    
    return test_metrics, all_preds, all_targets

def save_predictions_as_nc(predictions, original_dims, normalization_params, output_path, sample_indices=None):
    """将预测结果保存为nc文件，格式与目标数据一致"""
    print(f"Saving predictions to {output_path}...")
    
    # 获取原始维度信息
    target_shape = original_dims['target_shape']  # (time, window_row, window_col, win_row, win_col)
    time_coords = original_dims['time']
    window_start_row = original_dims['window_start_row']
    window_start_col = original_dims['window_start_col']
    sample_info = original_dims['sample_info']
    
    # 如果提供了样本索引，只使用这些样本
    if sample_indices is not None:
        # 创建一个全零数组用于存储预测结果
        pred_array = np.zeros(target_shape)
        pred_array[:] = np.nan  # 初始化为nan
        
        # 将预测结果放回正确的位置
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx < len(sample_info):
                info = sample_info[sample_idx]
                t = info['time_idx']
                wr = info['window_row_idx']
                wc = info['window_col_idx']
                pred_array[t, wr, wc] = predictions[i, 0]  # predictions shape: (n_samples, 1, 50, 50)
    else:
        # 将所有预测结果放回正确的位置
        pred_array = np.zeros(target_shape)
        pred_array[:] = np.nan
        
        for i, info in enumerate(sample_info):
            if i < len(predictions):
                t = info['time_idx']
                wr = info['window_row_idx']
                wc = info['window_col_idx']
                pred_array[t, wr, wc] = predictions[i, 0]
    
    # 创建xarray Dataset
    ds = xr.Dataset(
        {
            TARGET_VAR: (['time', 'window_row', 'window_col', 'win_row', 'win_col'], pred_array),
            'window_start_row': (['window_row', 'window_col'], window_start_row),
            'window_start_col': (['window_row', 'window_col'], window_start_col)
        },
        coords={
            'time': time_coords,
            'window_row': np.arange(pred_array.shape[1]),
            'window_col': np.arange(pred_array.shape[2]),
            'win_row': np.arange(pred_array.shape[3]),
            'win_col': np.arange(pred_array.shape[4])
        },
        attrs={
            'description': 'CNN model predictions for source apportionment',
            'variable': f'predicted_{TARGET_VAR}',
            'source_name': SOURCE_NAME,
            'emission_var': EMISSION_VAR,
            'model': 'SourceApportionmentCNN',
            'normalization_params': str(normalization_params)
        }
    )
    
    # 设置变量属性
    ds[TARGET_VAR].attrs = {
        '_FillValue': np.nan,
        'long_name': f'Predicted {TARGET_VAR} concentration',
        'units': 'ug/m3'
    }
    
    ds['window_start_row'].attrs = {
        'long_name': 'Starting row index in original grid (1-based)',
        'units': 'index'
    }
    
    ds['window_start_col'].attrs = {
        'long_name': 'Starting column index in original grid (1-based)',
        'units': 'index'
    }
    
    # 保存为nc文件
    ds.to_netcdf(output_path)
    print(f"Predictions saved to: {output_path}")
    
    return ds

def train_model():
    """训练模型，使用时间顺序划分数据集"""
    
    # 加载数据 X=输入特征，y=真实值，同时获取原始维度信息
    print(f"Loading and preprocessing data for {TARGET_VAR} source...")
    X, y, original_dims = load_and_preprocess_data()
    
    
    # 原始数据有744个小时，前576h作为训练验证集，后168h作为测试集
    print("Splitting data by time: first 576h for train/val, last 168h for test...")
    
    # 获取原始维度信息
    target_shape = original_dims['target_shape']  # (time, window_row, window_col, win_row, win_col)
    sample_info = original_dims['sample_info']
    
    total_hours = target_shape[0]
    train_val_hours = 576
    test_hours = 168
    
    print(f"Total hours: {total_hours}, Train/Val hours: {train_val_hours}, Test hours: {test_hours}")
    
    # 计算样本数
    samples_per_hour = target_shape[1] * target_shape[2]  # window_row * window_col
    total_train_val_samples = train_val_hours * samples_per_hour
    total_test_samples = test_hours * samples_per_hour
    
    print(f"Samples per hour: {samples_per_hour}")
    print(f"Train/Val samples: {total_train_val_samples}")
    print(f"Test samples: {total_test_samples}")
    
    # 按照时间顺序划分样本索引
    train_val_indices = list(range(total_train_val_samples))
    test_indices = list(range(total_train_val_samples, total_train_val_samples + total_test_samples))
    
    # 提取数据
    X_train_val = X[train_val_indices]
    y_train_val = y[train_val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Training/Validation set shape: {X_train_val.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 从训练验证集中划分训练集和验证集（使用随机划分）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # 及时清理内存
    del X, y, X_train_val, y_train_val
    gc.collect()
    
    # 标准化数据 - 仅使用训练集计算参数
    print("Normalizing data using training set parameters...")
    X_train_norm, X_val_norm, X_test_norm, normalization_params = normalize_data(
        X_train, X_val, X_test
    )
    
    # 处理目标数据的NaN值
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0)
    
    # 保存标准化参数到文件
    norm_params_path = output_dir / f"normalization_params_{TARGET_VAR}.pkl"
    with open(norm_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)
    print(f"Normalization parameters saved to: {norm_params_path}")
    
    # 及时清理内存
    del X_train, X_val, X_test
    gc.collect()
    
    # 创建数据加载器
    train_dataset = CAMxDataset(X_train_norm, y_train)
    val_dataset = CAMxDataset(X_val_norm, y_val)
    test_dataset = CAMxDataset(X_test_norm, y_test)
    
    # 由于网格变小，可以适当增加batch_size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 及时清理内存
    del X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test
    gc.collect()
    
    # 初始化模型
    model = SourceApportionmentCNN(input_channels=INPUT_CHANNELS, output_channels=1, 
                                  target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 训练参数
    num_epochs = 50
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training for {SOURCE_NAME} source...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 检查尺寸是否匹配
            if outputs.shape != batch_y.shape:
                outputs = nn.functional.interpolate(outputs, size=batch_y.shape[2:], mode='bilinear', align_corners=True)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_X.size(0)
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        all_preds = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # 检查尺寸是否匹配
                if outputs.shape != batch_y.shape:
                    outputs = nn.functional.interpolate(outputs, size=batch_y.shape[2:], mode='bilinear', align_corners=True)
                
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_X.size(0)
                
                # 收集预测和目标用于计算指标
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 计算验证集的评估指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_metrics = calculate_metrics(all_preds, all_targets, prefix="VAL_")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'target_var': TARGET_VAR,
                'source_name': SOURCE_NAME,
                'emission_var': EMISSION_VAR
            }
            # 保存最佳模型到output目录
            best_model_path = output_dir / f'best_{TARGET_VAR}_cnn.pth'
            torch.save(best_model_state, best_model_path)
            print(f"New best model saved with val_loss: {best_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # 及时清理内存
        del all_preds, all_targets
        gc.collect()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, '
              f'Val R2: {val_metrics.get("VAL_R2", np.nan):.4f}, '
              f'Val RMSE: {val_metrics.get("VAL_RMSE", np.nan):.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 保存最终模型到output目录
    final_model_state = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'target_var': TARGET_VAR,
        'source_name': SOURCE_NAME,
        'emission_var': EMISSION_VAR
    }
    final_model_path = output_dir / f'final_{TARGET_VAR}_cnn.pth'
    torch.save(final_model_state, final_model_path)
    
    # 绘制损失曲线
    loss_plot_path = output_dir / f'loss_curves_{TARGET_VAR}.png'
    plot_loss_curves(train_losses, val_losses, loss_plot_path)
    
    # 使用最佳模型在测试集上进行评估
    print(f"\nEvaluating best model on test set for {TARGET_VAR} source...")
    
    # 加载最佳模型
    best_model = SourceApportionmentCNN(input_channels=INPUT_CHANNELS, output_channels=1, 
                                       target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH)
    best_model.load_state_dict(best_model_state['model_state_dict'])
    best_model.to(device)
    
    test_metrics, test_predictions, test_targets = evaluate_on_test_set(
        best_model, test_loader, criterion, device, original_dims, normalization_params
    )
    
    # 保存测试集评估指标到CSV文件
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_path = output_dir / f'test_metrics_{TARGET_VAR}.csv'
    test_metrics_df.to_csv(test_metrics_path, index=False)
    print(f"Test metrics saved to: {test_metrics_path}")
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 保存测试集预测结果为nc文件
    test_predictions_path = output_dir / f'test_predictions_{TARGET_VAR}.nc'
    save_predictions_as_nc(
        predictions=test_predictions,
        original_dims=original_dims,
        normalization_params=normalization_params,
        output_path=test_predictions_path,
        sample_indices=test_indices  # 只保存测试集的预测
    )
    
      
    print(f"\nTraining for {TARGET_VAR} source completed and models saved!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # 保存训练历史
    train_history_path = output_dir / f'training_history_{TARGET_VAR}.pkl'
    with open(train_history_path, 'wb') as f:
        pickle.dump({
            'best_epoch': best_model_state['epoch'],
            'best_val_loss': best_model_state['val_loss'],
            'val_metrics': best_model_state.get('val_metrics', {}),
            'test_metrics': test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'normalization_params': normalization_params,
            'target_var': TARGET_VAR,
            'source_name': SOURCE_NAME,
            'emission_var': EMISSION_VAR,
            'train_val_indices': train_val_indices,
            'test_indices': test_indices
        }, f)
    
    # 清理内存
    del model, best_model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    gc.collect()
    
    return best_model_state, test_metrics

if __name__ == "__main__":
    # 训练模型
    best_model_info, test_metrics = train_model()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Target variable: {TARGET_VAR}")
    print(f"Source: {SOURCE_NAME}")
    print(f"Emission variable: {EMISSION_VAR}")
    print(f"All outputs saved to: {output_dir}")
    
    # 显示输出文件列表
    print("\nOutput files:")
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {file_path.name} ({file_size:.2f} MB)")
    
    print(f"\n{SOURCE_NAME} Source Test Set Performance Summary:")
    print(f"  IOA: {test_metrics.get('TEST_IOA', np.nan):.4f}")
    print(f"  R2: {test_metrics.get('TEST_R2', np.nan):.4f}")
    print(f"  RMSE: {test_metrics.get('TEST_RMSE', np.nan):.6f}")
    print(f"  MEAN_PRED: {test_metrics.get('TEST_MEAN_PRED', np.nan):.6f}")
    print(f"  MEAN_TARGET: {test_metrics.get('TEST_MEAN_TARGET', np.nan):.6f}")
    print(f"  MB: {test_metrics.get('TEST_MB', np.nan):.6f}")
    print("="*50)
