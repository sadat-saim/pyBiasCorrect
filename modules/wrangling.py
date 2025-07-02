import os
import pandas as pd
import xarray as xr


def wrangle(station_data_path, reanalysis_data_path, gcm_data_path):
    """
    path: path of the groundwater station CSV file
    """

    df = pd.read_excel(station_data_path)
    df = df[8:]
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header
    df.reset_index(drop=True)
    # set datetime index
    df['DATE TIME'] = pd.to_datetime(
        df['DATE TIME'], format='mixed',  dayfirst=True)
    df.sort_values(by='DATE TIME')
    df = df.drop('SL', axis=1)
    df = df.reset_index(drop=True)
    df.set_index('DATE TIME', inplace=True)
    # Make the selected columns to float
    columns = ['WATER TABLE (m)', 'RL PARAPET (m)',
               'PARAPET HEIGHT (m)', 'LATITUDE', 'LONGITUDE']
    df[columns] = df[columns].astype(float)
    df.sort_values(by='DATE TIME', inplace=True)

    lat = df.LATITUDE.iloc[0]
    lon = df.LONGITUDE.iloc[0]

    # ==============================================================================================================
    # Reanalysis Data
    # ==============================================================================================================
    # List all NetCDF files in the directory
    nc_files = [os.path.join(reanalysis_data_path, file) for file in os.listdir(
        reanalysis_data_path) if file.endswith('.nc')]

    # Initialize an empty list to store time series data
    time_series_data = []
    for file in nc_files:
        # Open the NetCDF file as an xarray dataset
        dataset = xr.open_dataset(file)
        # Extract the data for the specified latitude and longitude
        data_at_location = dataset.sel(lat=lat, lon=lon, method='nearest')
        # Extract the time series data
        time_series_data.append(data_at_location)
        # Close the dataset
        dataset.close()
    # Combine the time series data from all files
    combined_time_series_data = xr.concat(time_series_data, dim='time')
    df_ts = combined_time_series_data.to_dataframe()
    df_ts = df_ts.groupby(level=0).first()

    # ====================================================================================================
    # GCM Data
    # ====================================================================================================
    # List all NetCDF files in the directory
    nc_files_gcm = [os.path.join(gcm_data_path, file) for file in os.listdir(
        gcm_data_path) if file.endswith('.nc')]

    # Initialize an empty list to store time series data
    time_series_data_gcm = []
    for file in nc_files_gcm:
        # Open the NetCDF file as an xarray dataset
        dataset = xr.open_dataset(file)
        # Extract the data for the specified latitude and longitude
        data_at_location = dataset.sel(lat=lat, lon=lon, method='nearest')
        # Extract the time series data
        time_series_data_gcm.append(data_at_location)
        # Close the dataset
        dataset.close()
    # Combine the time series data from all files
    gcm_time_series_data = xr.concat(time_series_data_gcm, dim='time')
    df_gcm = gcm_time_series_data.to_dataframe()
    df_gcm = df_gcm.groupby(level=0).first()
    df_gcm.reset_index(inplace=True)
    df_gcm.set_index('time_bnds', inplace=True)
    df_gcm.drop(['time', 'lat', 'lon', 'lat_bnds', 'lon_bnds',
                'height', 'evspsbl'], axis=1, inplace=True)
    df_ts.drop(['lat', 'lon', 'pevpr'], axis=1, inplace=True)

    # scale the shum feature
    df_gcm['huss'] = df_gcm['huss']*1000
    # Rename the gcm features to the same name as reanalysis features
    df_gcm = df_gcm.rename(columns={'tas': 'air', 'tasmax': 'tmax', 'tasmin': 'tmin',
                           'pr': 'prate', 'ps': 'pres', 'mrros': 'runof', 'huss': 'shum'})
    df_gcm = df_gcm[['air', 'prate', 'pres', 'shum', 'tmax', 'tmin']]

    # returns a list containing [Groundwater Dataframe, Reanalysis Dataframe and GCM dataframe]
    dfs = [df, df_ts, df_gcm]

    for df in dfs:
        df.index = pd.to_datetime(df.index.astype(str))
    return dfs


def wrangle_gcm(gcm_data_path, lat, lon):
    """
    gcm_data_path : path to the gcm nc data
    lat: Latitude of the station
    lon: Longitude of the station
    return: a pandas.DataFrame
    """
    # ====================================================================================================
    # GCM Data
    # ====================================================================================================
    # List all NetCDF files in the directory
    nc_files_gcm = [os.path.join(gcm_data_path, file) for file in os.listdir(
        gcm_data_path) if file.endswith('.nc')]

    # Initialize an empty list to store time series data
    time_series_data_gcm = []
    for file in nc_files_gcm:
        # Open the NetCDF file as an xarray dataset
        dataset = xr.open_dataset(file)
        # Extract the data for the specified latitude and longitude
        data_at_location = dataset.sel(lat=lat, lon=lon, method='nearest')
        # Extract the time series data
        time_series_data_gcm.append(data_at_location)
        # Close the dataset
        dataset.close()
    # Combine the time series data from all files
    gcm_time_series_data = xr.concat(time_series_data_gcm, dim='time')
    df_gcm = gcm_time_series_data.to_dataframe()
    df_gcm = df_gcm.groupby(level=0).first()
    df_gcm.reset_index(inplace=True)
    df_gcm.set_index('time_bnds', inplace=True)
    df_gcm.drop(['time', 'lat', 'lon', 'lat_bnds', 'lon_bnds',
                'height', 'evspsbl'], axis=1, inplace=True)

    # scale the shum feature
    df_gcm['huss'] = df_gcm['huss']*1000
    # Rename the gcm features to the same name as reanalysis features
    df_gcm = df_gcm.rename(columns={'tas': 'air', 'tasmax': 'tmax', 'tasmin': 'tmin',
                           'pr': 'prate', 'ps': 'pres', 'mrros': 'runof', 'huss': 'shum'})
    df_gcm = df_gcm[['air', 'prate', 'pres', 'shum', 'tmax', 'tmin']]

    df_gcm.index = pd.to_datetime(df_gcm.index.astype(str))
    return df_gcm
