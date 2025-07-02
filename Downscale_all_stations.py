from modules import wrangle, wrangle_gcm, monthly_mean_imputer, add_seasons
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from modules import nested_bias_correction
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import multiprocessing
import pickle

os.chdir(os.path.expanduser(os.getcwd()))

# Path configurations
path_station = r"F:\Reanalysis Data\Monthly\Observed"
station_files = [os.path.join(path_station, file) for file in os.listdir(
    path_station) if file.endswith('.xlsx')]
path_reanalysis = r"F:\Reanalysis Data\Monthly\Reanalysis"
# ===================================================================
# Change Model Name From Here Before Running New Models
# ====================================================================
path_gcm = r"F:\Reanalysis Data\Monthly\GCM\INM CM5 0\historical"
output_dir = r"F:\Reanalysis Data\Monthly\Output\INM CM5 0"
model_dir = r"F:\Reanalysis Data\Monthly\Models\INM CM5 0"

# Create output and model directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def add_time_features(X):
    """
    Adds time-based features to a DataFrame with a DateTime index.

    Parameters:
    X (pd.DataFrame): DataFrame with a DateTime index.

    Returns:
    pd.DataFrame: Updated DataFrame with added time features.
    """
    X = X.copy()  # Avoid modifying the original DataFrame
    print(X.head())

    X.loc[:, 'quarter'] = X.index.quarter
    X.loc[:, 'month'] = X.index.month

    # Ensure add_seasons() is defined
    X.loc[:, 'season'] = add_seasons(X.index)

    # Adding cyclic features
    X.loc[:, 'month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
    X.loc[:, 'month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

    X.drop(columns='month', inplace=True)

    return X


def optimize_model(X_train, y_train, model_type='rf'):
    """
    Optimizes a model using BayesSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    model_type (str): Type of model to optimize ('rf', 'xgb', or 'svr').

    Returns:
    sklearn.pipeline.Pipeline: Optimized model pipeline.
    """
    # Base pipeline
    pipeline = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        StandardScaler()
    )

    # Define search spaces based on model type
    if model_type == 'lr':
        # For LinearRegression, you need to add the step correctly
        pipeline.steps.append(('model', LinearRegression()))
        pipeline.fit(X_train, y_train)
        return pipeline
    elif model_type == 'rf':
        pipeline.steps.append(
            ('model', RandomForestRegressor(random_state=42)))
        search_spaces = {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(3, 20),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 10),
            'model__max_features': Categorical(['sqrt', 'log2', None])
        }
    elif model_type == 'xgb':
        pipeline.steps.append(('model', XGBRegressor(random_state=42)))
        search_spaces = {
            'model__n_estimators': Integer(50, 300),
            'model__max_depth': Integer(3, 15),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__subsample': Real(0.5, 1.0),
            'model__colsample_bytree': Real(0.5, 1.0),
            'model__gamma': Real(0, 5),
            'model__min_child_weight': Integer(1, 10)
        }
    elif model_type == 'svr':
        pipeline.steps.append(('model', SVR()))
        search_spaces = {
            'model__C': Real(0.1, 100, prior='log-uniform'),
            'model__epsilon': Real(0.01, 1.0, prior='log-uniform'),
            'model__gamma': Real(0.001, 1.0, prior='log-uniform'),
            'model__kernel': Categorical(['rbf', 'linear', 'poly'])
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Configure BayesSearchCV
    opt = BayesSearchCV(
        pipeline,
        search_spaces,
        n_iter=20,  # Reduced for faster computation, increase for better results
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=0
    )

    # Run optimization
    opt.fit(X_train, y_train)

    return opt.best_estimator_


def downscale(station, reanalysis, gcm, model_type='rf'):
    """
    Performs downscaling using the specified model type.

    Parameters:
    station (str): Path to station data file.
    reanalysis (str): Path to reanalysis data.
    gcm (str): Path to GCM data.
    model_type (str): Type of model to use ('lr', 'rf', 'xgb', or 'svr').

    Returns:
    None: Results are saved to disk.
    """
    # Get the station file name for logging
    station_name = os.path.basename(station)
    print(f"Processing {station_name} with {model_type} model...")

    # Data preparation
    df = wrangle(station, reanalysis, gcm)
    imputed = monthly_mean_imputer(df[0]["WATER TABLE (m)"], "wtable")
    y = imputed
    X = add_time_features(df[1].loc[y.index[0]: y.index[-1]])
    X_gcm = add_time_features(df[-1])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Model optimization and training
    model = optimize_model(X_train, y_train, model_type)

    # Predictions and evaluation
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # Historical downscaling
    downscaled = model.predict(X_gcm[X_test.columns])
    downscaled = pd.Series(data=downscaled.reshape(-1),
                           index=X_gcm.index, name=f'downscaled_hist_{model_type}')

    # Bias correction
    y_observed = y[: downscaled.index[-1]]
    y_predicted = downscaled[y.index[0]:]

    # mbc = monthly_bias_correction(
    #     y_observed, y_predicted, variable_name=f"mbc_hist_{model_type}")

    # nbc = nested_bias_correction(
    #     y_observed, y_predicted, variable_name=f"nbc_hist_{model_type}")

    # SSP245 scenario downscaling
    ssp_path_245 = r"F:\Reanalysis Data\Monthly\GCM\INM CM5 0\ssp245"
    ssp_path_585 = r"F:\Reanalysis Data\Monthly\GCM\INM CM5 0\ssp585"
    lat = df[0].iloc[0]['LATITUDE']
    lon = df[0].iloc[0]['LONGITUDE']
    X_ssp_245 = add_time_features(wrangle_gcm(ssp_path_245, lat, lon))
    X_ssp_585 = add_time_features(wrangle_gcm(ssp_path_585, lat, lon))

    downscaled_ssp_245 = model.predict(X_ssp_245[X_test.columns])
    downscaled_ssp_245 = pd.Series(data=downscaled_ssp_245.reshape(-1),
                                   index=X_ssp_245.index, name=f'downscaled_ssp_245_{model_type}')

    # mbc_ssp_245 = monthly_bias_correction(
    #     y_observed, downscaled_ssp_245, variable_name=f"mbc_ssp_245_{model_type}")

    downscaled_ssp_585 = model.predict(X_ssp_585[X_test.columns])
    downscaled_ssp_585 = pd.Series(data=downscaled_ssp_585.reshape(-1),
                                   index=X_ssp_585.index, name=f'downscaled_ssp_585_{model_type}')

    # mbc_ssp_585 = monthly_bias_correction(
    #     y_observed, downscaled_ssp_585, variable_name=f"mbc_ssp_585_{model_type}")

    # Create results dataframe
    merged_df = y_observed.to_frame().join(
        y_predicted, how='outer').join(downscaled_ssp_245, how='outer').join(downscaled_ssp_585, how='outer')

    # Add model evaluation metrics
    merged_df[f"Train MSE ({model_type})"] = train_mse
    merged_df[f"Train R2 ({model_type})"] = train_r2
    merged_df[f"Test MSE ({model_type})"] = test_mse
    merged_df[f"Test R2 ({model_type})"] = test_r2

    # Extract and add metadata
    metadata = df[0].iloc[0].to_dict()
    keys_to_remove = [
        "OLD ID", "WATER TABLE (m)", "RL PARAPET (m)", "PARAPET HEIGHT (m)", "DEPTH (m)"]
    filtered_metadata = {key: value for key,
                         value in metadata.items() if key not in keys_to_remove}

    for key, value in filtered_metadata.items():
        merged_df[key] = value

    # Save results
    well_id = df[0].iloc[0]['WELL ID']
    merged_df.to_csv(os.path.join(
        output_dir, f"station_{well_id}_{model_type}.csv"))

    # Save model
    with open(os.path.join(model_dir, f"model_{well_id}_{model_type}.pkl"), 'wb') as f:
        pickle.dump(model, f)

    print(f"Station {well_id} done processing with {model_type} model...")

    return well_id, model_type, test_r2


def process_station(args):
    """
    Wrapper function to call downscale with multiple model types and handle errors.

    Parameters:
    args (tuple): (station_path, model_type)

    Returns:
    tuple: (well_id, model_type, test_r2) or (station_path, model_type, None) if error
    """
    station_path, model_type = args
    try:
        return downscale(station_path, path_reanalysis, path_gcm, model_type)
    except Exception as e:
        print(f"Error processing {station_path} with {model_type}: {e}")
        return os.path.basename(station_path), model_type, None


if __name__ == '__main__':
    # Create job arguments for all stations and model types
    # model_types = ['lr', 'rf', 'xgb', 'svr']
    model_types = ['lr', 'xgb']
    jobs = [(station, model_type)
            for station in station_files for model_type in model_types]

    # Number of workers
    num_workers = min(3, multiprocessing.cpu_count())

    # Run parallel processing
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_station, jobs), total=len(jobs)))

    # Collect and save summary results
    summary = pd.DataFrame(results, columns=['Station', 'Model', 'Test_R2'])
    summary.to_csv(os.path.join(
        output_dir, "model_performance_summary.csv"), index=False)

    # Identify best model for each station
    best_models = summary.loc[summary.groupby('Station')['Test_R2'].idxmax()]
    best_models.to_csv(os.path.join(
        output_dir, "best_models_summary.csv"), index=False)

    print("All processing complete!")
