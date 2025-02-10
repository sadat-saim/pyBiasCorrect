import pandas as pd
import numpy as np


def linear_scaling(y_observed, y_predicted, variable_name="linear_scaling_bias_corrected", method="additive"):
    """
    Corrects biases in the predicted values using the linear scaling method. 
    The method can be either additive or multiplicative.

    Note: Series must contain a DatetimeIndex.

    Args:
        y_observed (pandas.Series): Observed values of the variable with a DatetimeIndex.
        y_predicted (pandas.Series): Predicted values of the variable with a DatetimeIndex.
        variable_name (str, optional): Name for the corrected series. Defaults to "linear_scaling_bias_corrected".
        method (str, optional): Method for bias correction - "additive" or "multiplicative". Defaults to "additive".

    Returns:
        pandas.Series: Bias-corrected predicted values.

    Raises:
        ValueError: If the input method is not "additive" or "multiplicative".
        ValueError: If the input series do not have a DatetimeIndex.
    """

    # Ensure the inputs are pandas Series
    if not isinstance(y_observed, pd.Series) or not isinstance(y_predicted, pd.Series):
        raise ValueError(
            "Both y_observed and y_predicted must be pandas Series.")

    # Ensure the inputs have a DatetimeIndex
    if not (pd.api.types.is_datetime64_any_dtype(y_observed.index) and
            pd.api.types.is_datetime64_any_dtype(y_predicted.index)):
        raise ValueError(
            "Both y_observed and y_predicted must have a DatetimeIndex.")

    # Add month information for grouping
    observed_monthly_mean = y_observed.groupby(y_observed.index.month).mean()
    predicted_monthly_mean = y_predicted.groupby(
        y_predicted.index.month).mean()

    # Compute correction factor based on the selected method
    if method == "additive":
        correction_factor = observed_monthly_mean - predicted_monthly_mean
        corrected_values = y_predicted + \
            y_predicted.index.month.map(correction_factor)
    elif method == "multiplicative":
        correction_factor = observed_monthly_mean / predicted_monthly_mean
        corrected_values = y_predicted * \
            y_predicted.index.month.map(correction_factor)
    else:
        raise ValueError(
            "Invalid method. Choose either 'additive' or 'multiplicative'.")

    # Return the corrected series
    corrected_series = corrected_values.rename(variable_name)
    return corrected_series


def standardize_by_month(series):
    """
    Standardizes the values in the series by subtracting the monthly mean 
    and dividing by the monthly standard deviation.

    Args:
        series (pd.Series): A pandas Series with a datetime index.

    Returns:
        pd.Series: A pandas Series of standardized values.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series.")

    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("The Series must have a datetime index.")

    # Group by month and calculate mean and standard deviation
    monthly_mean = series.groupby(series.index.month).mean()
    monthly_std = series.groupby(series.index.month).std()

    # Standardize the values using the monthly mean and standard deviation
    standardized_values = (
        series - series.index.month.map(monthly_mean)) / series.index.month.map(monthly_std)

    return standardized_values


def destandardize_by_month(standardized_values, original_series):
    """
    Reverses the standardization process to get back the original values.

    Args:
        standardized_values (pd.Series): The standardized values.
        original_series (pd.Series): Original series used for standardization (provides mean and std).

    Returns:
        pd.Series: The destandardized values.
    """
    if not isinstance(standardized_values, pd.Series) or not isinstance(original_series, pd.Series):
        raise ValueError("Both inputs must be pandas Series.")

    if not pd.api.types.is_datetime64_any_dtype(original_series.index):
        raise ValueError("The Series must have a datetime index.")

    # Group by month and calculate mean and standard deviation
    monthly_mean = original_series.groupby(original_series.index.month).mean()
    monthly_std = original_series.groupby(original_series.index.month).std()

    # Destandardize the values using the monthly mean and standard deviation
    destandardized_values = (standardized_values * standardized_values.index.month.map(
        monthly_std)) + standardized_values.index.month.map(monthly_mean)

    return destandardized_values


def nested_bias_correction(y_observed, y_predicted, variable_name="nested_bias_corrected"):
    """
    Corrects biases in the predicted values using the nested bias correction (NBC) method. 

    Note: Series must contain a DatetimeIndex.
    Reference: https://doi.org/10.1061/(ASCE)HE.1943-5584.0000585
               https://doi.org/10.1029/2011WR010464

    Args:
        y_observed (pandas.Series): Observed values of the variable with a DatetimeIndex.
        y_predicted (pandas.Series): Predicted values of the variable with a DatetimeIndex.
        variable_name (str, optional): Name for the corrected series. Defaults to "nested_bias_corrected".

    Returns:
        pandas.Series: Bias-corrected predicted values.
    """

    # Step 1: Standardize the modelled monthly series
    # ===============================================
    y_prime = standardize_by_month(y_predicted)

    # Step 2: Monthly lag 1 autocorrelation correction
    # ================================================
    rho_predicted = y_predicted.groupby(
        y_predicted.index.month).apply(lambda x: x.autocorr(lag=1))
    rho_observed = y_observed.groupby(
        y_observed.index.month).apply(lambda x: x.autocorr(lag=1))

    unique_years = sorted(y_prime.index.year.unique(), reverse=True)
    unique_months = sorted(y_prime.index.month.unique(), reverse=True)

    # Initialized an empty series to store y_two_prime data
    y_two_prime = pd.Series(
        np.array(np.nan) * len(y_prime), index=y_prime.index)

    # returns an array of single value array([0.86417553])
    def y_two_prime_i_k(month, year):
        # Get current month's data
        y_prime_i_k = y_prime[(y_prime.index.month == month)
                              & (y_prime.index.year == year)]

        # If reached the base January of the starting year return the value of that month
        if (month == 1) and (year == min(unique_years)):
            return y_prime_i_k.values
        # If reached the January of a year use December of previous year to correct January
        if month == 1:
            y_prime_i_minus_one_k = y_prime[(y_prime.index.month == (
                12)) & (y_prime.index.year == year-1)]

            rest = np.sqrt(1 - (rho_observed[month]) ** 2) * (
                (y_prime_i_k.values - rho_predicted[month] * y_prime_i_minus_one_k.values) /
                np.sqrt(1 - (rho_predicted[month]) ** 2)
            )
            # Calculate y"_i_k for a given year which corrects monthly biases
            return rho_observed[month] * y_two_prime_i_k(12, year - 1) + rest

        else:
            # Get previous month's data y'_(i-1)
            y_prime_i_minus_one_k = y_prime[(y_prime.index.month == (
                month - 1)) & (y_prime.index.year == year)]
            # Calculate the corrected value
            rest = np.sqrt(1 - (rho_observed[month]) ** 2) * (
                (y_prime_i_k.values - rho_predicted[month] * y_prime_i_minus_one_k.values) /
                np.sqrt(1 - (rho_predicted[month]) ** 2)
            )
            # Calculate y"_i_k for a given year which corrects monthly biases
            return rho_observed[month] * y_two_prime_i_k(month - 1, year) + rest

    for year in unique_years:
        for month in unique_months:
            y_two_prime.loc[(y_two_prime.index.month == month) & (
                y_two_prime.index.year == year)] = y_two_prime_i_k(month, year)[0]

    # Step 3: Rescale the series using observed means and stds
    # ========================================================
    y_three_prime = destandardize_by_month(y_two_prime, y_observed)
    # Set negative values to zero
    y_three_prime[y_three_prime < 0] = 0

    # Step 4: Standardize the annual series
    # =====================================
    z_k = y_three_prime.resample('YS').mean()
    z_prime = (z_k - z_k.mean())/z_k.std()

    # Step 5: Yearly lag 1 autocorrelation correction
    # ================================================
    rho_predicted = y_predicted.resample('YS').mean().autocorr(lag=1)
    rho_observed = y_observed.resample('YS').mean().autocorr(lag=1)

    # Initialized an empty series to store z_two_prime data
    z_two_prime = pd.Series(
        np.array(np.nan) * len(z_prime), index=z_prime.index)

    # Nested iteration from last year to first year and last month to first month
    def z_two_prime_k(year):
        # Get current month's data
        z_prime_k = z_prime[z_prime.index.year == year]
        # Get previous month's data z_prime_k_minus_1
        z_prime_k_minus_one = z_prime[z_prime.index.year == (year-1)]

        if year == unique_years[-1]:
            # If it's Base-year, return current year's data
            return z_prime_k.values
        else:
            # Calculate the corrected value
            rest = np.sqrt(1 - (rho_observed) ** 2) * (
                (z_prime_k.values - rho_predicted * z_prime_k_minus_one.values) /
                np.sqrt(1 - (rho_predicted) ** 2)
            )
            # Calculate z"_k for a given year which corrects yearly biases
            # [2014, 2013, 2012, 2011, 2010] the below code returns unique_years[unique_years.index(2014) + 1]
            # output: 2013 // the index method returns the index of the current value in the array
            return rho_observed * z_two_prime_k(unique_years[unique_years.index(year) + 1]) + rest

    # Perform calculations
    for year in unique_years:
        # z_two_prime_k(year) returns an array of single value array([0.86417553])
        z_two_prime.loc[z_two_prime.index.year == year] = z_two_prime_k(year)[
            0]
    # resample the observed series by mean and find the mean and std of the resampled series
    z_three_prime = (z_two_prime * y_observed.resample('YS').mean().std()
                     ) + y_observed.resample('YS').mean().mean()
    # Set negative values to zero
    z_three_prime[z_three_prime < 0] = 0
    # Step 6: Apply yearly correction to the monthly corrected series
    # ===============================================================
    # transforms the index from 2014-01-01 to 2014 for mapping values
    z_ratio = (z_three_prime / z_k).groupby(z_k.index.year).mean()
    y_corrected = y_three_prime * y_three_prime.index.year.map(z_ratio)

    return y_corrected.rename(variable_name)
