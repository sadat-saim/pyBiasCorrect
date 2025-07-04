import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def scatter_with_identity_line(data_arr, observed_col, model_cols, figsize=(10, 12), rows=4, cols=3, method='mean'):
    """
    Plots scatterplots of observed vs. simulated values for multiple models based on a statistical method,
    with a 1:1 reference line and a custom legend.

    Parameters:
        data_arr (list of pd.DataFrame): List of DataFrames with observed and simulated data.
        observed_col (str): Column name for observed values.
        model_cols (list of str): List of model column names.
        figsize (tuple): Figure size.
        rows (int): Number of subplot rows.
        cols (int): Number of subplot columns.
        method (str): Aggregation method ('mean', 'std', 'lag1', 'lag0').
    """

    # Validate method
    supported_methods = ['mean', 'std', 'lag1', 'lag0']
    if method not in supported_methods:
        raise ValueError(f"Method must be one of {supported_methods}")

    fig, ax = plt.subplots(figsize=figsize, nrows=rows, ncols=cols)
    ax = ax.flatten()

    # Helper functions
    def lag1_autocorr(series):
        return series.autocorr(lag=1)

    def lag0_corr(obs, sim):
        return np.corrcoef(obs, sim)[0, 1]

    # Precompute values based on method
    for df in data_arr:
        for i, model in enumerate(model_cols):
            if method == 'mean':
                x_val = df[observed_col].mean()
                y_val = df[model].mean()
            elif method == 'std':
                x_val = df[observed_col].std()
                y_val = df[model].std()
            elif method == 'lag1':
                x_val = lag1_autocorr(df[observed_col])
                y_val = lag1_autocorr(df[model])
            elif method == 'lag0':
                x_val = df[observed_col].corr(df[model])
                y_val = x_val  # Since it's cross-correlation, x = y

            ax[i].scatter(x_val, y_val, alpha=0.7,
                          color='steelblue', edgecolor='black')

    # Add 1:1 lines
    for i, model in enumerate(model_cols):
        all_x = []
        all_y = []
        for df in data_arr:
            if method == 'mean':
                all_x.append(df[observed_col].mean())
                all_y.append(df[model].mean())
            elif method == 'std':
                all_x.append(df[observed_col].std())
                all_y.append(df[model].std())
            elif method == 'lag1':
                all_x.append(lag1_autocorr(df[observed_col]))
                all_y.append(lag1_autocorr(df[model]))
            elif method == 'lag0':
                cc = df[observed_col].corr(df[model])
                all_x.append(cc)
                all_y.append(cc)

        min_val = min(min(all_x), min(all_y))
        max_val = max(max(all_x), max(all_y))
        ax[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        ax[i].set_title(model)
        ax[i].set_xlabel('Observed')
        ax[i].set_ylabel('Simulated')

    # # Legend in the unused last subplot
    # total_subplots = rows * cols
    # if len(model_cols) < total_subplots:
    #     ax[-1].axis('off')
    #     dot_legend = mlines.Line2D([], [], color='steelblue', marker='o', linestyle='None',
    #                                markersize=8, markeredgecolor='black', label=method.upper() + ' Value')
    #     line_legend = mlines.Line2D(
    #         [], [], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    #     ax[-1].legend(handles=[dot_legend, line_legend],
    #                   loc='upper left', fontsize=12)
    # Create legend handles
    dot_legend = mlines.Line2D([], [], color='steelblue', marker='o', linestyle='None',
                               markersize=8, markeredgecolor='black', label=method.upper() + ' Value')
    line_legend = mlines.Line2D(
        [], [], color='red', linestyle='--', linewidth=2, label='1:1 Line')

    # Remove unused axes if needed
    total_subplots = rows * cols
    if len(model_cols) < total_subplots:
        for i in range(len(model_cols), total_subplots):
            ax[i].axis('off')  # turn off unused subplot

    # 🔽 Add a single legend below all subplots
    fig.legend(handles=[dot_legend, line_legend],
               loc='lower center',
               ncol=2,           # One row
               bbox_to_anchor=(0.5, -0.05),  # Adjust as needed
               fontsize=12)

    # 🔽 Adjust layout so the legend fits
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
