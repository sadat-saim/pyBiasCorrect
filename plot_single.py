import marimo

__generated_with = "0.11.12"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    sns.set_theme(style="darkgrid", palette="deep")
    return np, os, pd, plt, sns


@app.cell
def _(mo):
    mo.md("""# Plot a Single Station Data""")
    return


@app.cell
def _(pd):
    stat = pd.read_csv(r"F:\Reanalysis Data\Monthly\stats.csv")
    stat
    return (stat,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Observed VS Predicted
        Shows the statistics of observed vs predicted and bias corrected(MBC) by different models.
        """
    )
    return


@app.cell
def _(mo, stat):
    mo.hstack(
        [stat[["mean", "mean_lr", "mean_rf", "mean_svr", "mean_xgb"]].plot(xlabel="Stations", ylabel="Mean", title="Mean Water Table Level"),
        stat[["std", "std_lr", "std_rf", "std_svr", "std_xgb"]].plot(xlabel="Stations", ylabel="Std", title="Std. Water Table Level"),
        stat[["skew", "skew_lr", "skew_rf", "skew_svr", "skew_xgb"]].plot(xlabel="Stations", ylabel="Skew", title="Skew Water Table Level")]
    )
    return


@app.cell
def _(mo):
    mo.md("""### Visualizations for station **GT2947900**""")
    return


@app.cell
def _(os):
    processed_data = r"F:\Reanalysis Data\Monthly\Output\ACCESS ESM 15"
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
    return (
        file_name,
        file_path,
        model_type,
        parts,
        processed_data,
        station_files,
        station_groups,
        station_id,
    )


@app.cell
def _(pd, station_groups):
    for station, models in station_groups.items():
        print(station)
        df_arr = []
        for _, data_path in models.items():
            df_arr.append(pd.read_csv(data_path, index_col=0, parse_dates=True))
        df_combined = pd.concat(df_arr, axis=1)  # axis=1 means stacking columns
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        df_combined.to_csv(rf"F:\Reanalysis Data\Monthly\Combined\ASSESS ESM 15\{station}.csv")
    return data_path, df_arr, df_combined, models, station


@app.cell
def _():
    # df_combined = pd.concat(df_arr, axis=1)  # axis=1 means stacking columns
    # df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    # df_combined
    return


@app.cell
def _():
    # fig, ax = plt.subplots( nrows=2, figsize=(16, 8))
    # models = ['lr', 'svr', 'rf']
    # df_combined[['wtable', *[f"mbc_hist_{_}" for _ in models]]].dropna().plot(ax = ax[0], ylabel='Watertable', title='Observed VS Predicted Watertable Data')
    # df_combined[['wtable', *[f"mbc_hist_{_}" for _ in models]]].dropna().plot(kind='kde', ax = ax[1])
    return


@app.cell
def _():
    # df_combined[[f"mbc_ssp_245_{_}" for _ in models]].dropna().plot(kind='hist', bins=50, figsize=(15, 6), alpha=0.7)
    return


@app.cell
def _():
    # import altair as alt

    # # Reset index so that date becomes a column
    # df_reset = df_arr[0].reset_index()

    # # Extract month and year from the date if you want to display them on the axes
    # df_reset['Month'] = df_reset['index'].dt.month
    # df_reset['Year'] = df_reset['index'].dt.year

    # # Create the Altair chart with transposed axes
    # alt.Chart(df_reset).mark_rect().encode(
    #     alt.X('Year:N', title="Year"),  # Year on X-axis (transposed)
    #     alt.Y('Month:N', title="Month"),  # Month on Y-axis (transposed)
    #     alt.Color('mbc_ssp_245_lr:Q', title="Water Table Level"),  # Color represents water table levels
    #     tooltip=[
    #         alt.Tooltip('index:T', title="Date"),  # Tooltip for the date
    #         alt.Tooltip('Month:N', title="Month"),  # Tooltip for the month
    #         alt.Tooltip('Year:N', title="Year"),  # Tooltip for the year
    #         alt.Tooltip('mbc_ssp_245_lr:Q', title="Water Table Level")  # Tooltip for the water table level
    #     ]
    # ).properties(
    #     title=f"Water Table Projections Hitmap (SSP 245) for {df_reset['UPAZILA'].iloc[0]}"
    # ).configure_view(
    #     step=13,
    #     strokeWidth=0
    # ).configure_axis(
    #     domain=False
    # )
    return


@app.cell
def _():
    # # Create a bar chart for the water table levels (min and max) across months
    # bar = alt.Chart(df_reset).mark_bar(cornerRadius=10, height=10).encode(
    #     x=alt.X('min(mbc_ssp_245_lr):Q').scale(domain=[-1, 9]).title('Water Table Level (m)'),  # Adjust scale based on values
    #     x2='max(mbc_ssp_245_lr):Q',  # Max water table level for the range
    #     y=alt.Y('Month:O').title('Months')  # Month on the Y-axis
    # )

    # # Create text labels for min values (rounded to 2 decimal places)
    # text_min = alt.Chart(df_reset).mark_text(align='right', dx=-5).encode(
    #     x='min(mbc_ssp_245_lr):Q',
    #     y=alt.Y('Month:O'),
    #     text=alt.Text('min(mbc_ssp_245_lr):Q', format='.2f')  # Rounded to 2 decimal places
    # )

    # # Create text labels for max values (rounded to 2 decimal places)
    # text_max = alt.Chart(df_reset).mark_text(align='left', dx=5).encode(
    #     x='max(mbc_ssp_245_lr):Q',
    #     y=alt.Y('Month:O'),
    #     text=alt.Text('max(mbc_ssp_245_lr):Q', format='.2f')  # Rounded to 2 decimal places
    # )

    # # Combine the charts
    # (bar + text_min + text_max).properties(
    #     title=alt.Title(text='Water Table Variation by Month (SSP 245)', subtitle='Water table projections for SSP 245 (Model)'),
    #     width=900, height=300
    # )
    return


if __name__ == "__main__":
    app.run()
