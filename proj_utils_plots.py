from colorama import Style, Fore

# importing visualisation libraries and stylesheets
import matplotlib.pyplot as plt
import os
from IPython.display import HTML, display, FileLink
import seaborn as sns
import pandas as pd
import xgboost as xgb

from proj_configs import MPL_STYLE_FILE, PATH_OUT_VISUALS
import proj_utils

from proj_utils_logging import get_logger
logger = get_logger()

plt.style.use(MPL_STYLE_FILE)

class ColourStyling(object):
    blk = Style.BRIGHT + Fore.BLACK
    gld = Style.BRIGHT + Fore.YELLOW
    grn = Style.BRIGHT + Fore.GREEN
    red = Style.BRIGHT + Fore.RED
    blu = Style.BRIGHT + Fore.BLUE
    mgt = Style.BRIGHT + Fore.MAGENTA
    res = Style.RESET_ALL

custColour = ColourStyling()

# function to render colour coded print statements
def beautify(str_to_print: str, format_type: int = 0) -> str:
    color_map = {
        0: custColour.mgt,
        1: custColour.grn,
        2: custColour.gld,
        3: custColour.red
    }

    if format_type not in color_map:
        raise ValueError(f"format_type must be between 0 and {len(color_map) - 1}")

    return f"{color_map[format_type]}{str_to_print}{custColour.res}"

def display_plot_link(filename, base_dir='plots'):
    logger.debug("START ...")
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        display(FileLink(filepath))
    else:
        print(f"File {filepath} not found")
    logger.debug("... FINISH")

def plot_cardinality(cardinality_df, n_cat_threshold, threshold_used='ABS', type_of_cols='all', figsize=(10, 6)):
    logger.debug("START ...")
    stack_colours = ['#deffd4', '#ffffff']

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cardinality_df.iloc[:,:-1].plot.bar(
        x=cardinality_df.columns[0],
        stacked=True,
        ax=ax,
        linewidth=0.75,
        edgecolor="gray",
        color=stack_colours
    )
    ax.invert_xaxis()
    ax.set_xlabel('Column names')
    ax.set_ylabel('Percentage of rows')
    ax.set_title(f'Cardinality plot of {type_of_cols} columns')

    # Add a black dash for each bar to signify 'unique_pct' values
    for i, col_name in enumerate(cardinality_df.iloc[:,0]):
        unique_value = cardinality_df.loc[cardinality_df.iloc[:, 0] == col_name, cardinality_df.columns[-1]].values[0]
        ax.plot(i, unique_value, '_', markeredgecolor = 'black', markersize=10, markeredgewidth=1, label=('unique_pct' if i == 0 else None))

    if threshold_used == 'PCT':
        ax.axhline(y=n_cat_threshold, color='red', linestyle='-', linewidth=1, alpha=0.8, label=f'Threshold line at {n_cat_threshold}')

    # anchoring the legend box lower left corner to below X/Y coordinates scaled 0-to-1
    plt.legend(bbox_to_anchor=(1.0, 0))
    proj_utils.save_and_show_link(fig, f'plot_cardinality_{type_of_cols}_{proj_utils.get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")

def plot_numerical_distribution(df, features):
    logger.debug("START ...")
    if features is None or len(features) == 0:
        logger.debug("... FINISH")
        return

    # Calculate the number of rows and columns needed
    n_features = len(features)
    n_cols = 6  # Maximum cols to have in the plot grid
    n_rows = (n_features + n_cols - 1) // n_cols

    feat_idx = 0
    for row in range(n_rows):
        # Calculate number of columns for current row
        cols_in_row = min(n_cols, n_features - row * n_cols)

        # Create a figure and axes for the current row
        fig, axs = plt.subplots(2, cols_in_row,
                               gridspec_kw={"height_ratios": (0.7, 0.3)},
                               figsize=(4*cols_in_row, 4))

        # Make axs 2D if it's 1D (happens when cols_in_row = 1)
        if cols_in_row == 1:
            axs = axs.reshape(2, 1)

        for j in range(cols_in_row):
            axs_hist = axs[0, j]
            axs_box = axs[1, j]

            if feat_idx < len(features):
                current_feature = features[feat_idx]

                # Plot histogram
                axs_hist.hist(df[current_feature], color='lightgray',
                            edgecolor='gray', linewidth=0.5, bins=50)
                axs_hist.set_title(f'Plots for {current_feature}', fontsize=10)
                axs_hist.spines['top'].set_visible(False)
                axs_hist.spines['right'].set_visible(False)

                # Plot boxplot
                axs_box.boxplot(
                    df[current_feature],
                    vert=False,
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color': 'black'},
                    flierprops={
                        'marker': 'o',
                        'markerfacecolor': 'gray',
                        'markersize': 2
                    },
                    whiskerprops={'linewidth': 0.5},
                    boxprops={
                        'facecolor': 'lightgray',
                        'color': 'gray',
                        'linewidth': 1
                    },
                    capprops={'linewidth': 1}
                )

                axs_box.set(yticks=[])
                axs_box.spines['left'].set_visible(False)
                axs_box.spines['right'].set_visible(False)
                axs_box.spines['top'].set_visible(False)

                feat_idx += 1
            else:
                # Hide empty subplots
                axs_hist.set_visible(False)
                axs_box.set_visible(False)

        plt.tight_layout()
        plt.show()
    logger.debug("... FINISH")

def plot_categorical_distribution(df, features):
    logger.debug("START ...")
    if features is None or len(features) == 0:
        logger.debug("... FINISH")
        return

    # Calculate the number of rows and columns needed
    n_features = len(features)
    n_cols = 6  # Maximum cols to have in the plot grid
    n_rows = (n_features + n_cols - 1) // n_cols

    # Create a figure with a proper subplot grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axs = axs.ravel() # Flatten the array for easier indexing

    for idx, feature in enumerate(features):
        value_counts = df[feature].value_counts().sort_index()
        axs[idx].bar(range(len(value_counts)), value_counts.values, color='lightgray', edgecolor='gray', linewidth=0.5)

        # Set both the tick positions and labels
        axs[idx].set_xticks(range(len(value_counts)))  # Set tick positions
        axs[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')  # Set tick labels
        axs[idx].set_title(f'Distribution of {feature}', fontsize=10)

    # Hide empty subplots
    for idx in range(len(features), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    # plt.show()
    proj_utils.save_and_show_link(fig, f'plot_cat_distro_{n_features}feats_{proj_utils.get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")

def plot_relationship_to_target(df, features, target, trend_type=None):
    logger.debug("START ...")
    if features is None or len(features) == 0:
        logger.debug("... FINISH")
        return

    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axs = axs.ravel()  # Flatten the array for easier indexing

    for idx, feature in enumerate(features):
        # Group data by feature
        grouped_data = [group[target].values for name, group in df.groupby(feature)]

        # Create box plot
        axs[idx].boxplot(
            grouped_data,
            patch_artist=True,
            medianprops={
                'color': 'black'
            },
            flierprops={
                'marker': 'o',
                'markerfacecolor': 'gray',
                'markersize': 2
            },
            whiskerprops={
                'linewidth': 1
            },
            boxprops={
                'facecolor': 'lightgray',
                'color': 'gray',
                'linewidth': 1
            },
            capprops={
                'linewidth': 1
            }
        )

        # Set x-ticks with feature categories
        categories = sorted(df[feature].unique())
        axs[idx].set_xticklabels(categories, rotation=45, ha='right')
        axs[idx].set_title(f'Distribution of {target} by {feature}', fontsize=10)

        # Add trend line if specified
        if trend_type is not None:
            axs_twin_y = axs[idx].twinx()

            if trend_type == 'mean':
                trend_values = df.groupby(feature)[target].mean()
            elif trend_type == 'median':
                trend_values = df.groupby(feature)[target].median()

            # Plot trend line
            axs_twin_y.plot(
                range(1, len(categories) + 1),
                trend_values.values,
                color='red',
                marker='o',
                markersize=3,
                linewidth=1,
                alpha=0.6
            )

            # Set trend line axis properties
            axs_twin_y.tick_params(axis='y', colors='red')

    # Hide empty subplots
    for idx in range(len(features), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    # plt.show()
    proj_utils.save_and_show_link(fig, f'plot_relate_{n_features}feats_to_target_{target}_{proj_utils.get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")


def plot_metrics_snapshot(model_metrics, model_type=None):
    logger.debug("START ...")
    if model_metrics is None or len(model_metrics) == 0:
        logger.debug("... FINISH")
        return

    df_metrics = pd.DataFrame(model_metrics)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].plot(df_metrics['iteration'], df_metrics['train_mse'], color='green', label='Train MSE')
    axs[0].plot(df_metrics['iteration'], df_metrics['val_mse'], color='red', label='Val MSE')
    axs[0].plot(df_metrics['iteration'], df_metrics['test_mse'], color='blue', label='Test MSE')
    axs[0].set_title('MSE across Iterations', fontsize=10)
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Mean Squared Error')
    axs[0].legend()

    axs[1].plot(df_metrics['iteration'], df_metrics['train_r2'], color='green', label='Train R2')
    axs[1].plot(df_metrics['iteration'], df_metrics['val_r2'], color='red', label='Val R2')
    axs[1].plot(df_metrics['iteration'], df_metrics['test_r2'], color='blue', label='Test R2')
    axs[1].set_title('R2 across Iterations', fontsize=10)
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('R2 Score')
    axs[1].legend(bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()

    proj_utils.save_and_show_link(fig, f'plot_metrics_{proj_utils.get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")

def plot_correlation_with_target(df, target):
    """
    Plots the correlation of each variable in the dataframe with the 'demand' column.

    Args:
    - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the plot on a Jupyter window)
    """
    logger.debug("START ...")
    # Compute correlations between all variables and 'demand'
    correlations = df.corr()[target].drop(target).sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    # sns.set_style(
    #     "whitegrid", {"axes.facecolor": "#f6f3ec", "grid.linewidth": 2}
    # )  # Light grey background and thicker grid lines

    # Create a bar plot
    fig = plt.figure(figsize=(12, 8))
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title(f"Correlation with target: {target}", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    fig.patch.set_facecolor('#f6f3ec')  # Set figure background color
    ax = plt.gca()  # get current axes
    ax.set_facecolor('#f6f3ec')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    proj_utils.save_and_show_link(fig, f'plot_corr_with_{target}_{proj_utils.get_current_timestamp()}.png', dpi=600)

    # prevent matplotlib from displaying the chart every time we call this function
    plt.close(fig)
    logger.debug("... FINISH")

    return fig

def plot_residuals(preds_y, true_y, save_path=None):  # noqa: D417
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained XGBoost model.
    - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
    - valid_y (pd.Series): The true values for the validation set.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - None (Displays the residuals plot on a Jupyter window)
    """

    # Calculate residuals
    residuals = true_y - preds_y

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(true_y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Show the plot
    plt.close(fig)

    return fig

def plot_feature_importance(model, booster):
    """
    Plots feature importance for an XGBoost model.

    Args:
    - model: A trained XGBoost model

    Returns:
    - fig: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_type = "weight" if booster == "gblinear" else "gain"
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f"Feature Importance based on {importance_type}",
    )
    plt.tight_layout()
    plt.close(fig)

    return fig