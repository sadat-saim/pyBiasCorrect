import pandas as pd


def monthly_mean_imputer(series, variable_name="Imputed"):
    """Takes a pandas Series of daily, weekly or monthly frequency with missing values and returns a DataFrame with missing values 
    treated monthly dataframe using the monthly mean values. The Series must have a datetime index.

    Args:
        series (pandas.Series): A pandas Series with a datetime index.
        variable_name (str): The name of the variable (optional).

    Returns:
        pd.DataFrame: A dataframe with monthly mean frequency values.
    """
    # Ensure the input is a pandas Series
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")

    # Check if the index is of datetime type
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("Index of the series must be a datetime index.")

    # Resample to monthly frequency and compute the mean
    series_monthly = series.resample('MS').mean()
    series_monthly_mean = series_monthly.groupby(
        series_monthly.index.month).mean()

    # Find missing values in the series
    missing_values_index = series_monthly[series_monthly.isnull()].index

    # Impute missing values with the monthly mean

    for index in missing_values_index:
        series_monthly.loc[index == series_monthly.index] = series_monthly.loc[index ==
                                                                               series_monthly.index].index.month.map(series_monthly_mean)

    return series_monthly.rename(variable_name)
