from modules import wrangle,  wrangle_gcm, monthly_mean_imputer, add_seasons, monthly_bias_correction, nested_bias_correction
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import xarray as xr
import pandas as pd
import os
import multiprocessing

os.chdir(os.path.expanduser(os.getcwd()))

# my modules


path_station = r"F:\Reanalysis Data\Monthly\Observed"

station_files = [os.path.join(path_station, file) for file in os.listdir(
    path_station) if file.endswith('.xlsx')]

# Directory containing the reanalysis NetCDF files
path_reanalysis = r"F:\Reanalysis Data\Monthly\Reanalysis"
# Directory containing the gcm NetCDF files
path_gcm = r"F:\Reanalysis Data\Monthly\GCM\ACCESS ESM 15\historical"


def add_time_features(X):
    """
    Adds time-based features to a DataFrame with a DateTime index.

    Parameters:
    X (pd.DataFrame): DataFrame with a DateTime index.

    Returns:
    pd.DataFrame: Updated DataFrame with added time features.
    """
    X = X.copy()  # Avoid modifying the original DataFrame

    X.loc[:, 'quarter'] = X.index.quarter
    X.loc[:, 'month'] = X.index.month

    # Ensure add_seasons() is defined
    X.loc[:, 'season'] = add_seasons(X.index)

    # Adding cyclic features
    X.loc[:, 'month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X.loc[:, 'month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

    X.drop(columns='month', inplace=True)

    return X


def downscale(station, reanalysis, gcm):
    df = wrangle(station, reanalysis, gcm)

    imputed = monthly_mean_imputer(df[0]["WATER TABLE (m)"], "wtable")

    y = imputed

    X = add_time_features(df[1].loc[y.index[0]: y.index[-1]])
    X_gcm = add_time_features(df[-1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        StandardScaler(),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, model.predict(X_train))
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print("Train MSE and r2", train_mse, train_r2)
    print("Test MSE and r2", test_mse, test_r2)

    downscaled = model.predict(X_gcm[X_test.columns])
    downscaled = pd.Series(data=downscaled.reshape(-1),
                           index=X_gcm.index, name='downscaled_hist')

    y_observed = y[: downscaled.index[-1]]
    y_predicted = downscaled[y.index[0]:]

    mbc = monthly_bias_correction(
        y_observed, y_predicted, variable_name="mbc_hist")
    nbc = nested_bias_correction(
        y_observed, y_predicted, variable_name="nbc_hist")

    # ============================================================
    # SSP Path
    # ============================================================
    ssp_path = r"F:\Reanalysis Data\Monthly\GCM\ACCESS ESM 15\ssp245"

    lat = df[0].iloc[0]['LATITUDE']
    lon = df[0].iloc[0]['LONGITUDE']

    X_ssp_245 = add_time_features(wrangle_gcm(ssp_path, lat, lon))

    downscaled_ssp_245 = model.predict(X_ssp_245[X_test.columns])
    downscaled_ssp_245 = pd.Series(data=downscaled_ssp_245.reshape(-1),
                                   index=X_ssp_245.index, name='downscaled_ssp_245')

    mbc_ssp_245 = monthly_bias_correction(
        y_observed, downscaled_ssp_245, variable_name="mbc_ssp_245")

    # merged observed, downscaled, nbc, mbc and metadata
    merged_df = y_observed.to_frame().join(nbc, how='outer').join(mbc, how='outer').join(
        y_predicted, how='outer').join(downscaled_ssp_245, how='outer').join(mbc_ssp_245, how='outer')

    # Add model evaluation metrics
    merged_df["Train MSE"] = train_mse
    merged_df["Train R2"] = train_r2
    merged_df["Test MSE"] = test_mse
    merged_df["Test R2"] = test_r2

    # Extract metadata from the first row of df[0]
    metadata = df[0].iloc[0].to_dict()

    # Remove unwanted keys
    keys_to_remove = [
        "OLD ID", "WATER TABLE (m)", "RL PARAPET (m)", "PARAPET HEIGHT (m)", "DEPTH (m)"]
    filtered_metadata = {key: value for key,
                         value in metadata.items() if key not in keys_to_remove}

    # Add filtered metadata as new columns
    for key, value in filtered_metadata.items():
        merged_df[key] = value

    merged_df.to_csv(
        rf"F:\Reanalysis Data\Monthly\Output\station_{df[0].iloc[0]['WELL ID']}.csv")

    print(df[0].iloc[0]["WELL ID"], "Done processing ...")


# Use joblib or pickle to store multiple models in the same file

def process_station(station_path):
    """Wrapper function to call downscale and handle errors."""
    try:
        downscale(station_path, path_reanalysis, path_gcm)
    except Exception as e:
        print(f"Error processing {station_path}: {e}")


# On Windows, multiprocessing requires the script to be wrapped inside
# if __name__ == '__main__': to avoid process spawning issues.
if __name__ == '__main__':
    # Number of CPU cores to use, Since my cpu has 2 cores and 4 threads, I use 3 and leave 1 thread for other tasks
    # use   `multiprocessing.cpu_count()` to use all CPU cores with caution
    # Use up to 3 threads to keep system responsive
    num_workers = min(3, len(station_files))

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_station,
             station_files), total=len(station_files)))
