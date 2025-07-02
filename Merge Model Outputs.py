import pandas as pd
import os
from tqdm import tqdm

processed_data = r"F:\Reanalysis Data\Monthly\Output\INM CM5 0"

station_files = [os.path.join(processed_data, file) for file in os.listdir(
    processed_data) if file.endswith('.csv')]

# Group files by station ID
station_groups = {}
for file_path in station_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    if len(parts) >= 3:
        station_id, model_type = parts[1], parts[2].split('.')[0]
        station_groups.setdefault(station_id, {})[model_type] = file_path
# Merge model outputs
for station, models in tqdm(station_groups.items()):
    df_arr = []
    for _, data_path in models.items():
        df_arr.append(pd.read_csv(data_path, index_col=0, parse_dates=True))
    df_combined = pd.concat(df_arr, axis=1)  # axis=1 means stacking columns
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined = df_combined.sort_index(axis=1)
    df_combined.to_csv(
        rf"F:\Reanalysis Data\Monthly\Combined\INM CM5 0\{station}.csv")
