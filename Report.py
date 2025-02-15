import os
import numpy as np
import pandas as pd
from tqdm import tqdm


processed_data = r"F:\Reanalysis Data\Monthly\Output"

station_files = [os.path.join(processed_data, file) for file in os.listdir(
    processed_data) if file.endswith('.csv')]

df_arr = [pd.read_csv(station_file, index_col=0, parse_dates=True)
          for station_file in station_files]

stats_df = []

for df in tqdm(df_arr):
    params = {
        "district": df["DISTRICT"].iloc[0],
        "upazila": df["UPAZILA"].iloc[0],
        "mean_obs": df["wtable"].mean(),
        "std_obs": df["wtable"].std(),
        "skew_obs": df["wtable"].skew(),
        "mean_nbc": df["nested_bias_corrected"].mean(),
        "std_nbc": df["nested_bias_corrected"].std(),
        "skew_nbc": df["nested_bias_corrected"].skew(),
        "mean_mbc": df["monthly_bias_corrected"].mean(),
        "std_mbc": df["monthly_bias_corrected"].std(),
        "skew_mbc": df["monthly_bias_corrected"].skew(),
        "train_mse": df["Train MSE"].iloc[0],
        "train_r2": df["Train R2"].iloc[0],
        "test_mse": df["Test MSE"].iloc[0],
        "test_r2": df["Test R2"].iloc[0],
        "lat": df["LATITUDE"].iloc[0],
        "lon": df["LONGITUDE"].iloc[0]
    }

    stats_df.append(pd.DataFrame(params, index=[df.iloc[0]["WELL ID"]]))

stats_df = pd.concat(stats_df)

stats_df = stats_df.round(2)

stats_df.to_csv(r"F:\Reanalysis Data\Monthly\stats.csv")
