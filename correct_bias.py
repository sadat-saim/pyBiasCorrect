import numpy as np
import pandas as pd
from modules import monthly_bias_correction, nested_bias_correction
import os
import concurrent.futures
import traceback
from tqdm import tqdm

# Path to data
path = r"F:\Reanalysis Data\Monthly\Combined\ASSESS ESM 15"

# Get all station file paths
station_files = [os.path.join(path, file)
                 for file in os.listdir(path) if file.endswith('.csv')]

# Read all station CSVs into DataFrames
stations_df = [pd.read_csv(file, index_col=0, parse_dates=True)
               for file in station_files]


# Helper function to apply corrections
def apply_corrections(df):
    try:
        # Nested Bias Correction (NBC)
        correction_pairs_nbc = {
            'nbc_hist_lr': 'downscaled_hist_lr',
            'nbc_hist_rf': 'downscaled_hist_rf',
            'nbc_hist_svr': 'downscaled_hist_svr',
            'nbc_hist_xgb': 'downscaled_hist_xgb',
            'nbc_ssp_245_lr': 'downscaled_ssp_245_lr',
            'nbc_ssp_245_rf': 'downscaled_ssp_245_rf',
            'nbc_ssp_245_svr': 'downscaled_ssp_245_svr',
            'nbc_ssp_245_xgb': 'downscaled_ssp_245_xgb',
            'nbc_ssp_585_lr': 'downscaled_ssp_585_lr',
            'nbc_ssp_585_rf': 'downscaled_ssp_585_rf',
            'nbc_ssp_585_svr': 'downscaled_ssp_585_svr',
            'nbc_ssp_585_xgb': 'downscaled_ssp_585_xgb'
        }

        for new_col, model_col in correction_pairs_nbc.items():
            if model_col in df.columns:
                df[new_col] = nested_bias_correction(
                    df['wtable'].dropna(), df[model_col].dropna())
            else:
                print(
                    f"Column '{model_col}' not found for NBC correction. Skipping...")

        # Monthly Bias Correction (MBC)
        correction_pairs_mbc = {
            'mbc_hist_lr': 'downscaled_hist_lr',
            'mbc_hist_rf': 'downscaled_hist_rf',
            'mbc_hist_svr': 'downscaled_hist_svr',
            'mbc_hist_xgb': 'downscaled_hist_xgb',
            'mbc_ssp_245_lr': 'downscaled_ssp_245_lr',
            'mbc_ssp_245_rf': 'downscaled_ssp_245_rf',
            'mbc_ssp_245_svr': 'downscaled_ssp_245_svr',
            'mbc_ssp_245_xgb': 'downscaled_ssp_245_xgb',
            'mbc_ssp_585_lr': 'downscaled_ssp_585_lr',
            'mbc_ssp_585_rf': 'downscaled_ssp_585_rf',
            'mbc_ssp_585_svr': 'downscaled_ssp_585_svr',
            'mbc_ssp_585_xgb': 'downscaled_ssp_585_xgb'
        }

        for new_col, model_col in correction_pairs_mbc.items():
            if model_col in df.columns:
                df[new_col] = monthly_bias_correction(
                    df['wtable'].dropna(), df[model_col].dropna())
            else:
                print(
                    f"Column '{model_col}' not found for MBC correction. Skipping...")

    except Exception as e:
        print("Error processing DataFrame:")
        traceback.print_exc()

    return df


# Multithreading with tqdm progress bar
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    for result in tqdm(executor.map(apply_corrections, stations_df),
                       total=len(stations_df), desc="Applying Bias Corrections"):
        results.append(result)

# Optionally, save results
# corrected_path = os.path.join(path, 'corrected')
# os.makedirs(corrected_path, exist_ok=True)
# for i, df in enumerate(results):
#     filename = os.path.basename(station_files[i])
#     df.to_csv(os.path.join(corrected_path, filename))

print("✅ Bias correction completed for all stations.")
