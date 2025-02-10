import math
import matplotlib.pyplot as plt
import seaborn as sns


def plot_features_distributions(df_reanalysis, df_gcm):
    """
    Plots the distributions of features from two datasets (Reanalysis and GCM) 
    using density plots for comparison. Each feature is plotted in a subplot grid.

    Args:
        df_reanalysis (pd.DataFrame): DataFrame containing the reanalysis dataset.
        df_gcm (pd.DataFrame): DataFrame containing the GCM (General Circulation Model) dataset.

    Returns:
        None: The function displays the plot but does not return any value.
    """

    # Calculate the number of rows and columns for the subplot grid
    # Ensure all features are included
    rows = math.ceil(len(df_reanalysis.columns) / 2)
    columns = 2  # Fixed number of columns

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, columns, figsize=(15, 2 * rows))
    axs = axs.flatten()  # Flatten to easily index the subplots

    # Iterate through each feature in the reanalysis DataFrame
    for i, column in enumerate(df_reanalysis.columns):
        # Plot density of the current feature for Reanalysis and GCM datasets
        df_reanalysis[column].plot(
            kind="density", ax=axs[i], label="Reanalysis")
        df_gcm[column].plot(kind="density", ax=axs[i], label="GCM")

        # Label the x-axis with the column name
        axs[i].set_xlabel(column)
        axs[i].legend(["Reanalysis", "GCM"])  # Add legend for differentiation

    # Add a main title for the entire figure
    plt.suptitle("Distribution of Reanalysis and GCM data", fontsize=14)

    # Adjust spacing to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_residual_diagonistics(residuals_series, test_features_dataframe, show_title=False):
    """
    Creates diagnostic plots to evaluate the residuals of a regression model. 
    These include scatter plots of residuals vs each feature, a histogram of residuals, and a boxplot.

    Args:
        residuals_series (pd.Series): Residuals of the model (observed - predicted values).
        test_features_dataframe (pd.DataFrame): Features from the test dataset.

    Returns:
        None: Displays diagnostic plots but does not return any value.
    """

    # Number of predictors and figure setup
    num_predictors = test_features_dataframe.shape[1]
    fig, axs = plt.subplots(
        # Rows depend on predictors + additional diagnostics
        nrows=(num_predictors + 3) // 3,
        ncols=3,  # Fixed 3 columns
        figsize=(15, 3 * ((num_predictors + 3) // 3))
    )
    axs = axs.flatten()  # Flatten the axes array for easier iteration

    # Residuals vs Features scatter plots
    for i, column in enumerate(test_features_dataframe.columns):
        axs[i].scatter(test_features_dataframe[column],
                       residuals_series, alpha=0.7)
        axs[i].axhline(0, color='red', linestyle='--',
                       linewidth=1)  # Line at y=0
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Residuals')
        axs[i].grid()
        if show_title:
            axs[i].set_title(f'Residuals vs {column}')

    # Histogram of residuals with KDE
    sns.histplot(residuals_series, kde=True, bins=15, ax=axs[i + 1])
    if show_title:
        axs[i + 1].set_title("Residuals Histogram")
    axs[i + 1].set_xlabel("Residuals")

    # Boxplot of residuals
    sns.boxplot(x=residuals_series, ax=axs[i + 2], orient="h")
    if show_title:
        axs[i + 2].set_title("Residuals Boxplot")
    axs[i + 2].set_xlabel("Residuals")

    # Remove any empty subplots if they exist
    for j in range(i + 3, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_compare_time_series_and_boxplot(y_observed, y_predicted, variable_name="Variable [unit]", alt_legend="Downscaled"):
    """
    Plots a comparison of two time series (observed and predicted) along with their boxplots.

    Args:
        y_observed (pandas.Series): The observed time series data with a datetime index.
        y_predicted (pandas.Series): The predicted time series data with a datetime index.
        variable_name (str, optional): The name of the variable being plotted, including its unit. Defaults to "Variable [unit]".
        alt_legend (str, optional): The alternative legend for the predicted time series.

    Returns:
        None: Displays the plot directly.
    """
    fig, axs = plt.subplots(1, 2, figsize=(17, 5), width_ratios=[
                            0.8, 0.2], gridspec_kw={'wspace': 0.01})

    # First subplot: Time series comparison
    y_observed.loc[:y_predicted.index[-1]].plot(ax=axs[0], label=variable_name)
    y_predicted.loc[y_observed.index[0]:].plot(
        ax=axs[0], style=["--"], label=f"{alt_legend} {variable_name}")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(variable_name)
    axs[0].legend(loc='upper left')
    axs[0].set_title(f"Observed vs. {alt_legend} {variable_name}")

    # Second subplot: Boxplot comparison
    y_observed.rename(variable_name).plot(kind="box", ax=axs[1])
    y_predicted.rename(variable_name).plot(
        kind="box", ax=axs[1], color="orange")
    axs[1].set_xlabel(variable_name)
    axs[1].set_title(f"Boxplot of {variable_name}")
    axs[1].set_xticks([])
    # for label in axs[1].get_yticklabels():
    #     label.set_alpha(0)
    axs[1].yaxis.tick_right()
    # Adjust layout
    plt.tight_layout()
    plt.show()
