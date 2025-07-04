import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from scipy.interpolate import griddata
import xarray as xr
import rioxarray
from pykrige.ok import OrdinaryKriging
import os


def interpolated_clipped_map(df, variables, shapefile_path, buffer=2, lon_col='lon', lat_col='lat',
                             method='kriging', figsize=(15, 15), rows=1, cols=1,
                             cbar_label='Value', cmap='RdYlGn'):
    """
    Interpolates spatial data from a DataFrame using kriging or griddata and clips it to a shapefile.

    Args:
        df (pd.DataFrame): DataFrame with coordinates and variable values.
        variables (list): Variables to interpolate.
        shapefile_path (str): Path to shapefile for clipping.
        buffer (float): Padding around data points for interpolation grid.
        lon_col (str): Column name for longitude.
        lat_col (str): Column name for latitude.
        method (str): Interpolation method ('kriging' or 'griddata').
        figsize (tuple): Size of the figure.
        rows (int): Number of subplot rows.
        cols (int): Number of subplot columns.
        cbar_label (str): Colorbar label.
        cmap (str): Colormap.

    Raises:
        ValueError: If interpolation method is invalid or grid shape mismatches.
    """

    # Load and reproject shapefile
    shapefile = gpd.read_file(shapefile_path).to_crs(epsg=4326)

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # Interpolation grid
    lat_vals = np.linspace(gdf[lat_col].min() - buffer,
                           gdf[lat_col].max() + buffer, 100)
    lon_vals = np.linspace(gdf[lon_col].min() - buffer,
                           gdf[lon_col].max() + buffer, 100)
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

    interpolated_data = {}

    # Interpolation
    for var in variables:
        if var not in gdf.columns:
            print(f"Variable '{var}' not found in dataframe. Skipping.")
            continue

        if method == 'kriging':
            ok = OrdinaryKriging(
                gdf[lon_col].values, gdf[lat_col].values, gdf[var].values,
                variogram_model='linear', verbose=False, enable_plotting=False
            )
            z_grid, _ = ok.execute('grid', lon_vals, lat_vals)
            interpolated_data[var] = z_grid

        elif method == 'griddata':
            z_grid = griddata(
                (gdf[lon_col], gdf[lat_col]), gdf[var],
                (lon2d, lat2d), method='linear'
            )
            interpolated_data[var] = z_grid
        else:
            raise ValueError("Method must be 'kriging' or 'griddata'.")

    # Convert to xarray Dataset
    ds = xr.Dataset(
        {var: (['lat', 'lon'], interpolated_data[var]) for var in variables},
        coords={'lat': lat_vals, 'lon': lon_vals}
    )
    ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # Clip using shapefile
    clipped = ds.rio.clip(shapefile.geometry, shapefile.crs, drop=True)

    max_value = np.ceil(
        np.max([clipped[var].max().values for var in variables]))
    # Optional: min_value = np.floor(np.min([clipped[var].min().values for var in variables]))

    # Setup colorbar
    cbar_kwargs = {
        'orientation': 'horizontal',
        'pad': 0.05,
        'shrink': 0.5,
        # 'label': cbar_label,
    }

    # Plotting
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    if len(variables) > len(axes):
        raise ValueError(
            f"Subplot grid ({rows}x{cols}) is too small for {len(variables)} variables.")

    for i, var in enumerate(variables):
        plot = clipped[var].plot(
            ax=axes[i], cmap=cmap, vmin=0, vmax=max_value, cbar_kwargs=cbar_kwargs)
        plot.colorbar.ax.xaxis.set_label_position('top')
        plot.colorbar.ax.set_xlabel(cbar_label, fontsize=14)
        axes[i].set_title(var, fontsize=20)
        axes[i].set_axis_off()
    axes[-1].set_axis_off()

    plt.tight_layout()
    plt.show()


# shapefile_path = r"E:\Downloads\bgd_adm_bbs_20201113_shp\bgd_adm_bbs_20201113_SHP\bgd_admbnda_adm0_bbs_20201113.shp"
# data_path = r"F:\SUST Research Project\output_with_models"
# data_paths = [os.path.join(data_path, f)
#               for f in os.listdir(data_path) if f.endswith('.csv')]
# data_arr = [pd.read_csv(path, parse_dates=True, index_col="Date")
#             for path in data_paths]


# mean = []

# for df in data_arr:
#     data = {}
#     data['lat'] = df['lat'].iloc[0]
#     data['lon'] = df['lon'].iloc[0]
#     data['district'] = df['district'].iloc[0]
#     data['Station_Id'] = df['Station_Id'].iloc[0]
#     for var in df.columns[0:12]:
#         data[var] = df[var].groupby(df.index.year).max().mean()
#     mean.append(data)
# mean_df = pd.DataFrame(mean)


# interpolated_clipped_map(
#     df=mean_df,
#     variables=['MRI-ESM2-0', 'INM-CM5-0', 'INM-CM4-8',
#                'GISS-E2-2-G', 'CMCC-ESM2', 'CMCC-CM2-SR5'],
#     shapefile_path=shapefile_path,
#     cmap='RdYlGn',
#     figsize=(20, 12),
#     cbar_label='Precipitation (mm)',
#     method='kriging',
#     rows=2,
#     cols=3
# )
