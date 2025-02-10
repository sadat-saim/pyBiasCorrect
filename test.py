import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


def find_val(months, target_month):
    """Find indices for a specific month in a time series"""
    return np.where(months == target_month)[0]


def obs_stats(y, n):
    """
    Calculate observed statistics for nested bias correction

    Parameters:
    y : pd.Series - Observed time series with DateTimeIndex
    n : int - Number of grid cells (1 for single series)

    Returns:
    dict: Dictionary containing calculated statistics
    """
    # Convert to numpy array for easier handling
    y_values = y.values.reshape(-1, 1) if n == 1 else y.values
    months = y.index.month.values
    years = y.index.year.values

    # Initialize arrays
    y_rho = np.zeros((n, 12))
    y_mean = np.zeros((n, 12))
    y_sd = np.zeros((n, 12))

    # Monthly calculations
    for i in range(12):
        month = i + 1
        idx = find_val(months, month)
        if month == 1:
            idx = idx[1:]  # Skip first January like R's ind.i[-1]

        if len(idx) == 0:
            continue

        # Handle single vs multiple grid cells
        if n == 1:
            y_month = y_values[idx, 0]
            prev_month = y_values[idx-1, 0]
            y_rho[0, i] = np.corrcoef(prev_month, y_month)[
                0, 1] if len(y_month) > 1 else 0
            y_mean[0, i] = np.nanmean(y_month)
            y_sd[0, i] = np.nanstd(y_month)
        else:
            for j in range(n):
                y_month = y_values[idx, j]
                prev_month = y_values[idx-1, j]
                y_rho[j, i] = np.corrcoef(prev_month, y_month)[
                    0, 1] if len(y_month) > 1 else 0
                y_mean[j, i] = np.nanmean(y_month)
                y_sd[j, i] = np.nanstd(y_month)

    # Annual calculations
    annual = y.resample('YS').mean()
    z_rho = np.zeros(n)
    z_mean = np.zeros(n)
    z_sd = np.zeros(n)

    if n == 1:
        z_rho[0] = acf(annual, nlags=1, missing='conservative')[1]
        z_mean[0] = np.nanmean(annual)
        z_sd[0] = np.nanstd(annual)
    else:
        for j in range(n):
            z_rho[j] = acf(annual.iloc[:, j], nlags=1,
                           missing='conservative')[1]
            z_mean[j] = np.nanmean(annual.iloc[:, j])
            z_sd[j] = np.nanstd(annual.iloc[:, j])

    return {
        'mon_mean': y_mean,
        'mon_sd': y_sd,
        'mon_rho': y_rho,
        'yr_mean': z_mean,
        'yr_sd': z_sd,
        'yr_rho': z_rho
    }


def nest_mod_cal(n, y, obs_stats_dict, start_yr, end_yr):
    """
    Nested model calibration

    Parameters:
    n : int - Number of grid cells
    y : pd.Series - Model time series to calibrate
    obs_stats_dict : dict - Observed statistics from obs_stats
    start_yr : int - Start year of calibration period
    end_yr : int - End year of calibration period

    Returns:
    dict: Dictionary containing calibrated series and model statistics
    """
    # Initialize variables
    y_values = y.values.reshape(-1, 1) if n == 1 else y.values
    months = y.index.month.values
    nyrs = end_yr - start_yr + 1
    tol = 0.1

    # Handle zero months
    for i in range(12):
        month = i + 1
        idx = find_val(months, month)
        if n > 1:
            for j in range(n):
                if np.all(y_values[idx, j] == 0) and not np.any(np.isnan(y_values[idx, j])):
                    y_values[idx, j] = np.random.uniform(0, tol, len(idx))

    # Monthly standardization and autocorrelation correction
    y_std = np.zeros_like(y_values)
    y_mean = np.zeros((n, 12))
    y_sd = np.zeros((n, 12))
    m_rho_mod = np.zeros((n, 12))

    for i in range(12):
        month = i + 1
        idx = find_val(months, month)
        if len(idx) == 0:
            continue

        # Calculate model statistics
        if n == 1:
            y_month = y_values[idx, 0]
            y_mean[0, i] = np.nanmean(y_month)
            y_sd[0, i] = np.nanstd(y_month)
            y_std[idx, 0] = (y_month - y_mean[0, i]) / y_sd[0, i]
        else:
            for j in range(n):
                y_month = y_values[idx, j]
                y_mean[j, i] = np.nanmean(y_month)
                y_sd[j, i] = np.nanstd(y_month)
                y_std[idx, j] = (y_month - y_mean[j, i]) / y_sd[j, i]

        # Calculate autocorrelation
        if month == 1:
            idx = idx[1:]  # Skip first January
        if len(idx) == 0:
            continue

        prev_idx = idx - 1
        for j in range(n):
            if n == 1:
                valid = ~np.isnan(y_std[prev_idx, 0]
                                  ) & ~np.isnan(y_std[idx, 0])
                if np.sum(valid) > 1:
                    m_rho_mod[0, i] = np.corrcoef(
                        y_std[prev_idx[valid], 0], y_std[idx[valid], 0])[0, 1]
            else:
                valid = ~np.isnan(y_std[prev_idx, j]
                                  ) & ~np.isnan(y_std[idx, j])
                if np.sum(valid) > 1:
                    m_rho_mod[j, i] = np.corrcoef(
                        y_std[prev_idx[valid], j], y_std[idx[valid], j])[0, 1]

    # ... (remaining calibration steps for annual correction)

    # Return results (truncated for space)
    return {
        'gcm_cor': pd.Series(y_values.flatten(), index=y.index),
        'mod_stats': {
            'mon_mean': y_mean,
            'mon_sd': y_sd,
            'mon_rho': m_rho_mod,
            # ... other statistics
        }
    }


# Example usage
if __name__ == "__main__":
    # Load data (example)
    dates = pd.date_range('1990-01-01', '2010-12-31', freq='MS')
    obs_data = pd.Series(np.random.gamma(2, 2, len(dates)), index=dates)
    model_data = pd.Series(np.random.gamma(3, 3, len(dates)), index=dates)

    # Calculate observed statistics
    obs_stats_dict = obs_stats(obs_data, n=1)

    # Calibrate model
    calibrated = nest_mod_cal(n=1, y=model_data,
                              obs_stats_dict=obs_stats_dict,
                              start_yr=1990, end_yr=2010)

    print("Calibrated series head:")
    print(calibrated['gcm_cor'].head())
