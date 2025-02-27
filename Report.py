import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pprint
from sklearn.metrics import r2_score

# Define directory for processed data
processed_data = r"F:\Reanalysis Data\Monthly\Output\ACCESS ESM 15"

# Get all CSV files in the directory
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

# Pretty printer for debugging
pp = pprint.PrettyPrinter(indent=2, width=100, sort_dicts=False)

# Dictionary to store station statistics
all_station_data = {}

# Process each station
for station_id, models_path in tqdm(station_groups.items(), desc="Processing Stations"):
    try:
        base_df = pd.read_csv(models_path["lr"], index_col=0, parse_dates=True)

        # Extract station metadata
        valid_wtable = base_df["wtable"].replace(
            [np.inf, -np.inf], np.nan).dropna()
        station_record = {
            "station_id": station_id,
            "district": base_df["DISTRICT"].iloc[0],
            "upazila": base_df["UPAZILA"].iloc[0],
            "lat": base_df["LATITUDE"].iloc[0],
            "lon": base_df["LONGITUDE"].iloc[0],
            "mean": valid_wtable.mean(),
            "std": valid_wtable.std(),
            "skew": valid_wtable.skew(),
        }

        # Process each model
        for model, model_path in models_path.items():
            df = pd.read_csv(model_path, index_col=0, parse_dates=True)

            # Filter out invalid values
            valid_model_data = df[f"mbc_hist_{model}"].replace(
                [np.inf, -np.inf], np.nan).dropna()

            # Compute statistics for model
            station_record[f"mean_{model}"] = valid_model_data.mean()
            station_record[f"std_{model}"] = valid_model_data.std()
            station_record[f"skew_{model}"] = valid_model_data.skew()

            # Extract MSE and R2 scores
            for metric in ["Train MSE", "Test MSE", "Train R2", "Test R2"]:
                col_name = f"{metric} ({model})"
                if col_name in df.columns:
                    station_record[f"{metric.lower().replace(' ', '_')}_{model}"] = df[col_name].iloc[0]

            # Compute R2 score if valid
            try:
                valid_data = df[["wtable", f"mbc_hist_{model}"]].replace(
                    [np.inf, -np.inf], np.nan).dropna()
                if not valid_data.empty:
                    station_record[f"r2_{model}"] = r2_score(
                        valid_data["wtable"], valid_data[f"mbc_hist_{model}"])
                else:
                    station_record[f"r2_{model}"] = np.nan
            except ValueError as err:
                print(
                    f"Error computing R2 for station {station_id}, model {model}: {err}")
                station_record[f"r2_{model}"] = np.nan
            except Exception as err:
                print(
                    f"Unexpected error for R2 calculation in station {station_id}, model {model}: {err}")
                station_record[f"r2_{model}"] = np.nan

        # Store station data
        all_station_data[station_id] = station_record
    except Exception as e:
        print(f"Skipping station {station_id} due to error: {e}")

# Convert to DataFrame
all_station_df = pd.DataFrame.from_dict(all_station_data, orient='index')
all_station_df = all_station_df.sort_index(axis=1)

# Round numeric columns for better readability
numeric_columns = all_station_df.select_dtypes(
    include=['float64', 'float32']).columns
all_station_df[numeric_columns] = all_station_df[numeric_columns].round(2)

# Save results
output_path = r"F:\Reanalysis Data\Monthly\stats.csv"
all_station_df.to_csv(output_path)
print(f"Processed station statistics saved to {output_path}")
