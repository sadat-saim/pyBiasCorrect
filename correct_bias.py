import numpy as np
import pandas as pd
from modules import monthly_bias_correction
import os
import concurrent.futures
import traceback
from tqdm import tqdm

# Path to data
path = r"F:\Reanalysis Data\Monthly\Combined\INM CM5 0"

# Get all station file paths
station_files = [os.path.join(path, file)
                 for file in os.listdir(path) if file.endswith('.csv')]

# Read all station CSVs into DataFrames
stations_df = [pd.read_csv(file, index_col=0, parse_dates=True)
               for file in station_files]


# Helper function to apply corrections
def apply_corrections(df):
    try:
        # Nested Bias Correction (mbc)
        correction_pairs_mbc = {
            'mbc_hist_lr': 'downscaled_hist_lr',
            'mbc_hist_xgb': 'downscaled_hist_xgb',
            'mbc_ssp_245_lr': 'downscaled_ssp_245_lr',
            'mbc_ssp_245_xgb': 'downscaled_ssp_245_xgb',
            'mbc_ssp_585_lr': 'downscaled_ssp_585_lr',
            'mbc_ssp_585_xgb': 'downscaled_ssp_585_xgb'
        }

        for new_col, model_col in correction_pairs_mbc.items():
            if model_col in df.columns:
                df[new_col] = monthly_bias_correction(
                    df['wtable'].dropna(), df[model_col].dropna())
            else:
                print(
                    f"Column '{model_col}' not found for mbc correction. Skipping...")

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
corrected_path = os.path.join(path, 'corrected')
os.makedirs(corrected_path, exist_ok=True)
for i, df in enumerate(results):
    filename = os.path.basename(station_files[i])
    df.to_csv(os.path.join(corrected_path, filename))

print("✅ Bias correction completed for all stations.")
