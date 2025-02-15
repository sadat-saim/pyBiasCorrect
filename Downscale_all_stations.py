from modules import wrangle, monthly_mean_imputer, add_seasons, monthly_bias_correction, nested_bias_correction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import xarray as xr
import pandas as pd
import os
os.chdir(os.path.expanduser(os.getcwd()))

# my modules


path_station = r"F:\Reanalysis Data\Monthly\Observed"

station_files = [os.path.join(path_station, file) for file in os.listdir(
    path_station) if file.endswith('.xlsx')]

# Directory containing the reanalysis NetCDF files
path_reanalysis = r"F:\Reanalysis Data\Monthly\Reanalysis"
# Directory containing the gcm NetCDF files
path_gcm = r"F:\Reanalysis Data\Monthly\GCM\ACCESS ESM 15\historical"


def downscale(station, reanalysis, gcm):
    df = wrangle(path_station, path_reanalysis, path_gcm)

    print(df[0].iloc[0]["WELL ID"], "Started processing ...")

    imputed = monthly_mean_imputer(df[0]["WATER TABLE (m)"], "wtable")

    y = imputed

    X = df[1].loc[y.index[0]: y.index[-1]].copy()
    X.loc[:, 'month'] = X.index.month
    X.loc[:, 'quarter'] = X.index.quarter
    # X.loc[:, 'year'] = X.index.year

    X.loc[:, 'season'] = add_seasons(X.index)
    # Adding cyclic features (optional)
    X.loc[:, 'month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X.loc[:, 'month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

    X_gcm = df[-1]
    X_gcm.loc[:, 'month'] = X_gcm.index.month
    # X_gcm = gcm.loc[y.index[-1]:].copy()
    X_gcm.loc[:, 'quarter'] = X_gcm.index.quarter
    X_gcm.loc[:, 'season'] = add_seasons(X_gcm.index)
    # Adding cyclic features (optional)
    X_gcm.loc[:, 'month_sin'] = np.sin(2 * np.pi * X_gcm['month'] / 12)
    X_gcm.loc[:, 'month_cos'] = np.cos(2 * np.pi * X_gcm['month'] / 12)
    # X_gcm.loc[:, 'year'] = X_gcm.index.year
    X.drop(columns='month', inplace=True)
    X_gcm.drop(columns='month', inplace=True)

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
                           index=X_gcm.index, name='wtable')
    mbc = monthly_bias_correction(
        y[: downscaled.index[-1]], downscaled[y.index[0]:])
    nbc = nested_bias_correction(
        y[: downscaled.index[-1]], downscaled[y.index[0]:])

    merged_df = imputed.to_frame().join(nbc, how='outer').join(mbc, how='outer').join(
        downscaled[y.index[0]:], how='outer', rsuffix='_downscaled').dropna()

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


# downscale(path_station, path_reanalysis, path_gcm)

for station_path in station_files:
    downscale(station_path, path_reanalysis, path_gcm)
