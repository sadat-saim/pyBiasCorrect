import pandas as pd


def add_seasons(date_index):
    """
    Takes a pandas DatetimeIndex and returns a pandas Series of seasons.

    Args:
        date_index (pandas.DatetimeIndex): A pandas DatetimeIndex.

    Returns:
        pandas.Series: A pandas Series containing the corresponding seasons.
    """
    # Validate input
    if not isinstance(date_index, pd.DatetimeIndex):
        raise TypeError("Input must be a pandas DatetimeIndex.")

    # Extract the month from the DatetimeIndex and map it to seasons
    month = date_index.month
    season_mapping = {
        1: "Winter", 2: "Winter", 12: "Late Autumn",
        3: "Spring", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
        9: "Autumn", 10: "Autumn", 11: "Late Autumn"
    }
    return pd.Series(month, index=date_index, name="Seasons").map(season_mapping)
