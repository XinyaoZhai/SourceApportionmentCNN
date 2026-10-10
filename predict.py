import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import gc
import time
from tqdm import tqdm

# ==================== 配置参数（与训练时保持一致） ====================
TARGET_VAR = 'PN4_G1'
SOURCE_NAME = 'FERTILIZER'
EMISSION_VAR = 'NH3'
MET_VARIABLES = ['PBLH', 'PSFC', 'RAIN', 'RH', 'T2', 'WD', 'WS']
INPUT_CHANNELS = 9
TARGET_HEIGHT = 50
TARGET_WIDTH = 50

# 数据路径（与训练时相同）
BASE_DATA_DIR = "./data_50"
CAMX_DATA_DIR = f"{BASE_DATA_DIR}/CAMX"
MEIC_DATA_DIR = f"{BASE_DATA_DIR}/MEIC"
WRF_DATA_DIR = f"{BASE_DATA_DIR}/WRF"

# 输出目录（存放训练产出）
output_dir = Path("./train_output")

# 模型文件路径（最佳模型）
model_path = output_dir / f'best_{TARGET_VAR}_cnn.pth'
norm_params_path = output_dir / f'normalization_params_{TARGET_VAR}.pkl'

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==================== 数据集类（与训练时相同） ====================
class CAMxDataset(Dataset):
    def __init__(self, input_data, target_data=None):
        self.input_data = input_data
        self.target_data = target_data  # 可以为None（仅预测时不需要target）

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_tensor = torch.FloatTensor(self.input_data[idx])
        if self.target_data is not None:
            target_tensor = torch.FloatTensor(self.target_data[idx])
            return input_tensor, target_tensor
        else:
            return input_tensor

# ==================== 模型定义（与训练时相同） ====================
class SourceApportionmentCNN(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, output_channels=1,
                 target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):
        super(SourceApportionmentCNN, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.ReLU()
        )

        self.final_adjust = nn.Sequential(
            nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if x.shape[2] != self.target_height or x.shape[3] != self.target_width:
            x = self.final_adjust(x)
        return x

# ==================== 数据加载函数（仅加载测试集） ====================
def load_test_data():
    """加载测试集数据（最后168小时），返回输入数组和原始维度信息"""
    print("Loading test data...")

    camx_psat_path = f"{CAMX_DATA_DIR}/CN03km_{TARGET_VAR}_rowcol_202401.nc"
    meic_path = f"{MEIC_DATA_DIR}/MEIC202401_{EMISSION_VAR}_{SOURCE_NAME}_rowcol.nc"
    camx_target_path = f"{CAMX_DATA_DIR}/CN03km_{TARGET_VAR}_22sector_rowcol_202401.nc"

    # 读取目标数据以获取维度信息
    with xr.open_dataset(camx_target_path) as ds:
        target_data = ds[TARGET_VAR].values
        time_coords = ds.time.values
        window_start_row = ds.window_start_row.values
        window_start_col = ds.window_start_col.values
        target_shape = target_data.shape  # (time, window_row, window_col, win_row, win_col)

    total_hours = target_shape[0]
    train_val_hours = 576
    test_hours = 168
    assert total_hours == 744, "数据总小时数不是744"

    # 加载所有输入数据（完整时间序列）
    input_datasets = []

    # PSAT
    with xr.open_dataset(camx_psat_path) as ds:
        psat_data = ds[TARGET_VAR].values
        input_datasets.append(psat_data)

    # 排放
    with xr.open_dataset(meic_path) as ds:
        emission = ds[EMISSION_VAR].values
        input_datasets.append(emission)

    # 气象
    for var in MET_VARIABLES:
        wrf_path = f"{WRF_DATA_DIR}/CN03_WRF_{var}_202401.nc"
        with xr.open_dataset(wrf_path) as ds:
            met_data = ds[var].values
            input_datasets.append(met_data)

    # 构建完整样本数组（所有时间、窗口）
    num_samples = target_shape[0] * target_shape[1] * target_shape[2]
    num_channels = len(input_datasets)
    height, width = target_shape[3], target_shape[4]

    input_combined = np.zeros((num_samples, num_channels, height, width), dtype=np.float32)
    sample_info = []  # 记录每个样本的时间、窗口索引

    sample_idx = 0
    for t in range(target_shape[0]):
        for wr in range(target_shape[1]):
            for wc in range(target_shape[2]):
                for ch, data in enumerate(input_datasets):
                    input_combined[sample_idx, ch] = data[t, wr, wc]
                sample_info.append({
                    'time_idx': t,
                    'window_row_idx': wr,
                    'window_col_idx': wc,
                    'time': time_coords[t] if t < len(time_coords) else None
                })
                sample_idx += 1

    # 计算测试集样本索引（最后168小时）
    samples_per_hour = target_shape[1] * target_shape[2]
    total_train_val_samples = train_val_hours * samples_per_hour
    test_indices = list(range(total_train_val_samples, total_train_val_samples + test_hours * samples_per_hour))

    X_test = input_combined[test_indices]

    # 记录原始维度信息（用于后续保存nc文件）
    original_dims = {
        'time': time_coords,
        'window_start_row': window_start_row,
        'window_start_col': window_start_col,
        'target_shape': target_shape,
        'sample_info': [sample_info[i] for i in test_indices]
    }

    return X_test, original_dims

# ==================== 标准化函数（使用保存的参数） ====================
def normalize_test_data(X_test, norm_params):
    """使用训练集的参数标准化测试数据"""
    X_test_norm = np.zeros_like(X_test)
    for i in range(X_test.shape[1]):
        channel_data = np.nan_to_num(X_test[:, i], nan=0.0)
        mean_val = norm_params[f'channel_{i}_mean']
        std_val = norm_params[f'channel_{i}_std']
        if std_val > 0:
            X_test_norm[:, i] = (channel_data - mean_val) / std_val
        else:
            X_test_norm[:, i] = channel_data - mean_val
    return X_test_norm

# ==================== 主程序：预测并计时 ====================
def main():
    # 1. 加载标准化参数
    print("Loading normalization parameters...")
    with open(norm_params_path, 'rb') as f:
        norm_params = pickle.load(f)

    # 2. 加载测试数据
    X_test_raw, original_dims = load_test_data()
    print(f"Test data shape: {X_test_raw.shape}")

    # 3. 标准化
    X_test = normalize_test_data(X_test_raw, norm_params)
    del X_test_raw
    gc.collect()

    # 4. 创建 DataLoader（batch_size 可根据显存调整，这里用32）
    test_dataset = CAMxDataset(X_test, target_data=None)  # 无标签
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 5. 加载模型
    print("Loading best model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SourceApportionmentCNN(input_channels=INPUT_CHANNELS, output_channels=1,
                                   target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 6. 预测并计时
    print("Starting prediction on test set (168 hours)...")
    start_time = time.time()

    all_preds = []
    with torch.no_grad():
        for batch_X in tqdm(test_loader, desc="Predicting"):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            # 确保输出尺寸正确
            if outputs.shape[2:] != (TARGET_HEIGHT, TARGET_WIDTH):
                outputs = nn.functional.interpolate(outputs, size=(TARGET_HEIGHT, TARGET_WIDTH),
                                                    mode='bilinear', align_corners=True)
            all_preds.append(outputs.cpu().numpy())

    end_time = time.time()
    total_time = end_time - start_time

    # 7. 统计结果
    all_preds = np.concatenate(all_preds, axis=0)  # (n_samples, 1, 50, 50)
    n_samples = len(test_dataset)
    samples_per_hour = original_dims['target_shape'][1] * original_dims['target_shape'][2]  # 窗口数
    n_hours = n_samples // samples_per_hour
    assert n_hours == 168, f"期望168小时，实际{n_hours}小时"

    print("\n" + "="*50)
    print(f"Prediction completed in {total_time:.2f} seconds")
    print(f"Total samples predicted: {n_samples}")
    print(f"Hours predicted: {n_hours} (7 days)")
    print(f"Average time per sample: {total_time / n_samples * 1000:.4f} ms")
    print(f"Average time per hour: {total_time / n_hours:.2f} seconds")
    print(f"Throughput: {n_samples / total_time:.2f} samples/second")
    print("="*50)

    # 可选：保存预测结果为nc文件（与训练脚本中的函数类似）
    # 这里简单保存，需要可自行取消注释
    # from train_script import save_predictions_as_nc   # 如果可导入
    # save_predictions_as_nc(all_preds, original_dims, norm_params, "test_prediction_7days.nc")

    # 清理
    del model, test_loader, test_dataset, X_test
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
