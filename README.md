# SourceApportionmentCNN
PyTorch implementation of Convolutional Neural Network (CNN) for air quality source apportionment.

## 1. Software & Environment Requirements
The code is developed and tested in Python. The required dependencies are:
- `torch` (PyTorch, CUDA recommended for GPU acceleration)
- `numpy`
- `xarray` & `netCDF4` (for reading and saving `.nc` files)
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`

Install all required packages via pip:
`pip install torch numpy xarray netCDF4 scikit-learn pandas matplotlib tqdm`

## 2. Data Preparation & Directory Structure
Place the preprocessed NetCDF (`.nc`) datasets into the `./data_50` directory following this structure:

- `train.py`: Main model training and validation script
- `predict.py`: Model inference and timing benchmark script on the test set
- `data_50/`
  - `CAMX/`: `CN03km_PN4_G1_rowcol_202401.nc`, `CN03km_PN4_G1_22sector_rowcol_202401.nc`
  - `MEIC/`: `MEIC202401_NH3_FERTILIZER_rowcol.nc`
  - `WRF/`: `CN03_WRF_[PBLH/PSFC/RAIN/RH/T2/WD/WS]_202401.nc`
- `train_output/`: Automatically generated directory for saving model weights and outputs

## 3. Usage
- **Step 1 (Model Training):** Run `python train.py` to train the CNN model, save the best weights, and evaluate metrics.
- **Step 2 (Inference & Benchmark):** Run `python predict.py` to load the best model (`best_PN4_G1_cnn.pth`) and evaluate inference speed on the 168-hour test set.

## 4. Outputs
All outputs are saved to `./train_output`:
- `best_PN4_G1_cnn.pth` & `final_PN4_G1_cnn.pth`: Saved model weights.
- `normalization_params_PN4_G1.pkl`: Training set normalization parameters.
- `loss_curves_PN4_G1.png`: Training and validation loss curves.
- `test_metrics_PN4_G1.csv`: Evaluation metrics (MSE, RMSE, MAE, R2, MAPE, IOA, MB) on the test set.
- `test_predictions_PN4_G1.nc`: Gridded model predictions on the test set in NetCDF format.
