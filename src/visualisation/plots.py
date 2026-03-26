"""
plots.py — Visualisation and plotting utility module.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as m_ticker
import os
from IPython.display import display, FileLink
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.config.settings import MPL_STYLE_FILE, CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS, CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT
from src.utils.io import save_and_show_link, get_current_timestamp
from src.utils.logging import get_logger

logger = get_logger()

plt.style.use(MPL_STYLE_FILE)

class ColourStyling:
    BOLD = '\033[1m'
    blk = BOLD + '\033[30m'
    gld = BOLD + '\033[33m'
    grn = BOLD + '\033[32m'
    red = BOLD + '\033[31m'
    blu = BOLD + '\033[34m'
    mgt = BOLD + '\033[35m'
    res = '\033[0m'

custColour = ColourStyling()

# function to render colour coded print statements
def beautify(str_to_print: str, format_type: int = 0) -> str:
    """
    Wrap a string in ANSI colour codes for console/notebook output.

    Parameters
    ----------
    str_to_print : str
    format_type : int
        Colour index: 0=magenta, 1=green, 2=gold, 3=red. Defaults to 0.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If format_type is outside 0-3.
    """
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
    """
    Show a clickable FileLink for a saved plot in the notebook.

    Parameters
    ----------
    filename : str
    base_dir : str
        Directory containing the file. Defaults to 'plots'.
    """
    logger.debug("START ...")
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        display(FileLink(filepath))
    else:
        print(f"File {filepath} not found")
    logger.debug("... FINISH")

def plot_cardinality(cardinality_df, n_cat_threshold, threshold_used=CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS, type_of_cols='all', figsize=(10, 6)):
    """
    Bar chart of null, non-null, and unique value percentages per column.

    Parameters
    ----------
    cardinality_df : pd.DataFrame
        Output of get_cardinality_df.
    n_cat_threshold : int or float
        Threshold value used when classifying columns.
    threshold_used : str
        'ABS' or 'PCT'. A horizontal threshold line is drawn only for 'PCT'.
    type_of_cols : str
        Label used in the plot title. Defaults to 'all'.
    figsize : tuple
        Figure dimensions. Defaults to (10, 6).
    """
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
    for i, _ in enumerate(cardinality_df.iloc[:, 0]):
        unique_value = cardinality_df.iloc[i, -1]
        ax.plot(i, unique_value, '_', markeredgecolor='black', markersize=10,
                markeredgewidth=1, label=('unique_pct' if i == 0 else None))

    # Threshold line only applicable for pct-based threshold (abs can't map to 0-1 scale)
    if threshold_used == CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT:
        ax.axhline(y=n_cat_threshold, color='red', linestyle='-', linewidth=1, alpha=0.8, label=f'Threshold line at {n_cat_threshold}')

    # anchoring the legend box lower left corner to below X/Y coordinates scaled 0-to-1
    ax.legend(bbox_to_anchor=(1.0, 0), loc='lower left')
    save_and_show_link(fig, f'plot_cardinality_{type_of_cols}_{get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")


def plot_numerical_distribution(df, features):
    """
    Histogram and boxplot pairs for each numerical feature, paginated by rows.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
        Numerical column names to plot.
    """
    logger.debug("START ...")
    if not features:
        logger.debug("... FINISH")
        return

    n_features = len(features)
    n_cols = 6
    n_rows = (n_features + n_cols - 1) // n_cols

    feat_idx = 0
    for row in range(n_rows):
        cols_in_row = min(n_cols, n_features - feat_idx)

        fig, axs = plt.subplots(
            2, cols_in_row,
            gridspec_kw={"height_ratios": (0.7, 0.3)},
            figsize=(4 * cols_in_row, 4)
        )

        # Ensure axs is always 2D (edge case: single column)
        if cols_in_row == 1:
            axs = axs.reshape(2, 1)

        for j in range(cols_in_row):
            current_feature = features[feat_idx]
            axs_hist = axs[0, j]
            axs_box = axs[1, j]

            # Histogram
            axs_hist.hist(df[current_feature], color='lightgray',
                          edgecolor='gray', linewidth=0.5, bins=50)
            axs_hist.set_title(f'Plots for {current_feature}', fontsize=10)
            axs_hist.spines['top'].set_visible(False)
            axs_hist.spines['right'].set_visible(False)

            # Boxplot (horizontal, aligned beneath histogram)
            axs_box.boxplot(
                df[current_feature],
                vert=False,
                widths=0.7,
                patch_artist=True,
                medianprops={'color': 'black'},
                flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 2},
                whiskerprops={'linewidth': 0.5},
                boxprops={'facecolor': 'lightgray', 'color': 'gray', 'linewidth': 1},
                capprops={'linewidth': 1}
            )
            axs_box.set_yticks([])
            axs_box.spines['left'].set_visible(False)
            axs_box.spines['right'].set_visible(False)
            axs_box.spines['top'].set_visible(False)

            feat_idx += 1

        fig.tight_layout()
        save_and_show_link(fig,
                           f'plot_num_distro_{n_features}feats_{row + 1}-of-{n_rows}_{get_current_timestamp()}.png')
        plt.close(fig)

    logger.debug("... FINISH")


def plot_categorical_distribution(df, features):
    """
    Value count bar charts for each categorical feature, paginated by rows.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
        Categorical column names to plot.
    """
    logger.debug("START ...")
    if not features:
        logger.debug("... FINISH")
        return

    n_features = len(features)
    n_cols = 6
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axs = axs.ravel()

    for idx, feature in enumerate(features):
        value_counts = df[feature].value_counts().sort_index()
        axs[idx].bar(
            range(len(value_counts)), value_counts.values,
            color='lightgray', edgecolor='gray', linewidth=0.5
        )
        axs[idx].set_xticks(range(len(value_counts)))
        axs[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axs[idx].set_title(f'Distribution of {feature}', fontsize=10)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)

    # Hide unused subplot slots in the last row
    for idx in range(n_features, len(axs)):
        axs[idx].set_visible(False)

    fig.tight_layout()
    save_and_show_link(fig, f'plot_cat_distro_{n_features}feats_{get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")


def plot_relationship_to_target(df, features, target, trend_type=None):
    """
    Boxplots of the target across each feature's categories, with an optional trend overlay.

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
        Categorical column names to plot against target.
    target : str
        Name of the target column.
    trend_type : str, optional
        Overlay line per category: 'mean' or 'median'. Defaults to None.
    """
    logger.debug("START ...")
    if not features:
        logger.debug("... FINISH")
        return

    if trend_type is not None and trend_type not in ('mean', 'median'):
        raise ValueError(f"trend_type must be 'mean', 'median', or None — got '{trend_type}'")

    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = axs.ravel()

    for idx, feature in enumerate(features):
        categories = sorted(df[feature].unique())
        grouped_data = [group[target].values for _, group in df.groupby(feature)]

        axs[idx].boxplot(
            grouped_data,
            patch_artist=True,
            medianprops={'color': 'black'},
            flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 2},
            whiskerprops={'linewidth': 1},
            boxprops={'facecolor': 'lightgray', 'color': 'gray', 'linewidth': 1},
            capprops={'linewidth': 1}
        )

        # Boxplot uses 1-based positions — set ticks explicitly to avoid FixedLocator warning
        axs[idx].set_xticks(range(1, len(categories) + 1))
        axs[idx].set_xticklabels(categories, rotation=45, ha='right')
        axs[idx].set_title(f'Distribution of {target} by {feature}', fontsize=10)

        if trend_type is not None:
            trend_values = (
                df.groupby(feature)[target].mean() if trend_type == 'mean'
                else df.groupby(feature)[target].median()
            )
            axs_twin_y = axs[idx].twinx()
            axs_twin_y.plot(
                range(1, len(categories) + 1),
                trend_values.values,
                color='red', marker='o', markersize=3, linewidth=1, alpha=0.6
            )
            axs_twin_y.tick_params(axis='y', colors='red')

    for idx in range(n_features, len(axs)):
        axs[idx].set_visible(False)

    fig.tight_layout()
    save_and_show_link(fig,
                       f'plot_relate_{n_features}feats_to_target_{target}_{get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")


def plot_metrics_snapshot(model_metrics, model_type=None):
    """
    Line plots of MSE and R2 across training iterations.

    Parameters
    ----------
    model_metrics : list of dict
        Each entry needs 'iteration', 'train_mse', 'val_mse', 'test_mse',
        'train_r2', 'val_r2', 'test_r2' keys.
    model_type : str, optional
        Reserved for future use; not currently used in the plot.
    """
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

    save_and_show_link(fig, f'plot_metrics_{get_current_timestamp()}.png')
    plt.close(fig)
    logger.debug("... FINISH")

def _compute_corr_scores(df, target, method='pearson'):
    """
    Compute Pearson or Spearman correlation of all features against the target.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    method : str
        'pearson' or 'spearman'. Defaults to 'pearson'.

    Returns
    -------
    pd.Series
        Sorted ascending, with the target column dropped.
    """
    return (
      df.corrwith(df[target], method=method)
        .drop(target)
        .sort_values()
    )

def _compute_mi_scores(df, target, random_state=43):
    """
    Compute Mutual Information scores of all features against the target.

    Numeric columns with NaNs are median-imputed before scoring, as required
    by mutual_info_regression.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    random_state : int
        Defaults to 43.

    Returns
    -------
    pd.Series
        Sorted ascending.
    """
    features = df.drop(columns=[target])
    y        = df[target].values
    X        = features.apply(
      lambda col: col.fillna(col.median()) if col.dtype.kind in 'fiu' else col
    )
    scores = mutual_info_regression(X, y, random_state=random_state)
    return pd.Series(scores, index=features.columns).sort_values()


def _render_corr_hbar(ax, series, title, xlabel, cmap, norm, show_zero_line=False):
    """
    Draw a colour-mapped horizontal bar chart on the given axes.

    Handles spines, grid, axis labels, and an optional zero reference line.
    Shared across all correlation plot functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    series : pd.Series
        Values to plot; index used as y-axis labels.
    title : str
    xlabel : str
    cmap : matplotlib colormap
    norm : matplotlib.colors.Normalize
    show_zero_line : bool
        If True, draws a vertical line at x=0 and sets xlim to (-1, 1).
    """
    color_mapped = [cmap(norm(v)) for v in series.values]
    ax.barh(series.index, series.values, color=color_mapped)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Variable", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='x')
    if show_zero_line:
      ax.set_xlim(-1, 1)
      ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    for spine in ax.spines.values():
      spine.set_visible(False)

def plot_correlation_with_target(df, target):
    """Plots Pearson correlation of each feature with the target."""
    logger.debug("START ...")
    scores = _compute_corr_scores(df, target, method='pearson')
    cmap   = sns.diverging_palette(10, 130, as_cmap=True)
    norm   = plt.Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    _render_corr_hbar(ax, scores, f"Pearson Correlation with target: {target}",
               "Pearson Correlation Coefficient", cmap, norm, show_zero_line=True)
    fig.tight_layout()
    save_and_show_link(fig, f'plot_corr_with_{target}_{get_current_timestamp()}.png', dpi=600)
    logger.debug("... FINISH")
    return fig

def plot_spearman_correlation_with_target(df, target):
    """Plots Spearman rank correlation of each feature with the target."""
    logger.debug("START ...")
    scores = _compute_corr_scores(df, target, method='spearman')
    cmap   = sns.diverging_palette(10, 130, as_cmap=True)
    norm   = plt.Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    _render_corr_hbar(ax, scores, f"Spearman Correlation with target: {target}",
               "Spearman Correlation Coefficient", cmap, norm, show_zero_line=True)
    fig.tight_layout()
    save_and_show_link(fig, f'plot_spearman_corr_with_{target}_{get_current_timestamp()}.png', dpi=600)
    logger.debug("... FINISH")
    return fig

def plot_mutual_information_with_target(df, target, random_state=43):
    """Plots Mutual Information score of each feature with the target."""
    logger.debug("START ...")
    scores = _compute_mi_scores(df, target, random_state)
    norm   = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap   = plt.cm.YlGn  # Sequential — MI has no sign, only magnitude

    fig, ax = plt.subplots(figsize=(12, 8))
    _render_corr_hbar(ax, scores, f"Mutual Information with target: {target}",
               "Mutual Information Score", cmap, norm, show_zero_line=False)
    fig.tight_layout()
    save_and_show_link(fig, f'plot_mi_with_{target}_{get_current_timestamp()}.png', dpi=600)
    logger.debug("... FINISH")
    return fig

def plot_feature_relevance_comparison(df, target, random_state=43):
    """
    Side-by-side Pearson, Spearman, and Mutual Information charts for all features.

    All three panels share the same y-axis feature order (sorted by MI score)
    so cross-panel comparison is straightforward.

    Parameters
    ----------
    df : pd.DataFrame
        Must include all feature columns and the target.
    target : str
        Name of the target column.
    random_state : int
        Seed for MI estimation. Defaults to 43.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.debug("START ...")

    pearson  = _compute_corr_scores(df, target, method='pearson')
    spearman = _compute_corr_scores(df, target, method='spearman')
    mi       = _compute_mi_scores(df, target, random_state)

    # Align all three to MI's ranking order (ascending = bottom-to-top on barh)
    shared_order = mi.sort_values(ascending=True).index
    pearson      = pearson.reindex(shared_order)
    spearman     = spearman.reindex(shared_order)
    mi           = mi.reindex(shared_order)

    div_cmap  = sns.diverging_palette(10, 130, as_cmap=True)
    div_norm  = plt.Normalize(vmin=-1, vmax=1)
    mi_norm   = plt.Normalize(vmin=mi.min(), vmax=mi.max())

    fig, axes = plt.subplots(1, 3, figsize=(36, max(8, len(shared_order) * 0.3)))

    _render_corr_hbar(axes[0], pearson,  f"Pearson vs {target}",
               "Pearson r", div_cmap, div_norm, show_zero_line=True)
    _render_corr_hbar(axes[1], spearman, f"Spearman vs {target}",
               "Spearman ρ", div_cmap, div_norm, show_zero_line=True)
    _render_corr_hbar(axes[2], mi,       f"Mutual Information vs {target}",
               "MI Score", plt.cm.YlGn, mi_norm, show_zero_line=False)

    # Only the leftmost panel needs a y-axis label — suppress on panels 2 and 3
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(f"Feature Relevance Comparison — target: {target}",
               fontsize=20, y=1.01)
    fig.tight_layout()
    save_and_show_link(fig, f'plot_relevance_comparison_{target}_{get_current_timestamp()}.png', dpi=600)

    logger.debug("... FINISH")
    return fig

def plot_residuals(preds_y, true_y):
    """
    Scatter plot of residuals (true minus predicted) against true values.

    Parameters
    ----------
    preds_y : array-like
        Predicted values.
    true_y : array-like
        True target values.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.debug("START ...")

    # Residuals: positive = under-prediction, negative = over-prediction
    residuals = true_y - preds_y

    # Scope seaborn style to this plot only — avoids overriding the global
    # matplotlib stylesheet for later plots in the session.
    with sns.axes_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}):
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(true_y, residuals, color="blue", alpha=0.5)

        # Reference line at zero — perfect predictions lie on this line
        plt.axhline(y=0, color="r", linestyle="-")

        plt.title("Residuals vs True Values", fontsize=18)
        plt.xlabel("True Values", fontsize=16)
        plt.ylabel("Residuals", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y")
        plt.tight_layout()

    plt.close(fig)

    logger.debug("... FINISH")
    return fig

def plot_feature_importance(pipeline, top_n: int = 30, importance_type: str = "gain"):
    """
    Horizontal bar chart of the top-N XGBoost feature importances.

    Extracts feature names from the pipeline's preprocessor step and maps
    them to the booster's internal feature indices.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with 'preprocessor' and 'regressor' steps.
    top_n : int
        Number of top features to show. Defaults to 30.
    importance_type : str
        'gain' or 'weight'. Passed to XGBoost's get_score. Defaults to 'gain'.

    Returns
    -------
    matplotlib.figure.Figure
    """
    preprocessor = pipeline.named_steps['preprocessor']
    regressor    = pipeline.named_steps['regressor']

    feature_names = preprocessor.get_feature_names_out()

    # get_feature_importance returns a dict keyed by internal booster feature names
    booster = regressor.get_booster()
    scores  = booster.get_score(importance_type=importance_type)

    # Booster uses 'f0', 'f1', ... — map back to real names
    importance_series = pd.Series(
        {feature_names[int(k[1:])]: v for k, v in scores.items()}
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(importance_series.index, importance_series.values,
            color='steelblue', edgecolor='none')

    ax.set_title(f"Top {top_n} Feature Importances ({importance_type})", fontsize=14)
    ax.set_xlabel(f"Importance ({importance_type})", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_and_show_link(fig, f'plot_feat_importance_top{top_n}_{get_current_timestamp()}.png')
    plt.close(fig)

    return fig


def plot_model_comparison(
        model_metrics: dict,
        plot_type: str = 'bar',
) -> plt.Figure:
    """
    Compare cross-validated MAE across models using a bar chart or box plot.

    Parameters
    ----------
    model_metrics : dict
        Keys are model display names. Values are metrics dicts with 'cv_mae_mean'
        and 'cv_mae_std' for bar mode, or 'cv_mae' (list of per-fold values) for
        box mode.
    plot_type : str
        'bar' for a bar chart with error bars, 'box' for a box plot. Defaults to 'bar'.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If plot_type is not 'bar' or 'box'.
    """
    logger.debug("START ...")
    logger.info(
        f"[plot_model_comparison] models={list(model_metrics.keys())} | plot_type={plot_type}"
    )

    if plot_type not in ('bar', 'box'):
        raise ValueError(
            f"[plot_model_comparison] plot_type must be 'bar' or 'box', got '{plot_type}'"
        )

    model_names = list(model_metrics.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    if plot_type == 'bar':
        mae_means = [model_metrics[m]['cv_mae_mean'] for m in model_names]
        mae_stds = [model_metrics[m]['cv_mae_std'] for m in model_names]

        bars = ax.bar(
            model_names, mae_means, yerr=mae_stds, capsize=5,
            edgecolor='black', linewidth=0.5,
        )

        # Annotate each bar with its MAE value for quick reading
        for bar, mean in zip(bars, mae_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.03,
                f'${mean:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
            )

        ax.set_ylim(0, max(mae_means) * 1.25)

    else:  # box
        # Validate that per-fold data is present in every metrics dict
        missing = [m for m in model_names if 'cv_mae_folds' not in model_metrics[m]]
        if missing:
            raise ValueError(
                f"[plot_model_comparison] 'cv_mae_folds' missing from: {missing}. "
                f"Add it to the return dict of the relevant run_* / tune_* functions."
            )

        fold_data = [model_metrics[m]['cv_mae_folds'] for m in model_names]

        bp = ax.boxplot(
            fold_data, labels=model_names, patch_artist=True,
            medianprops=dict(color='darkorange', linewidth=2),
        )
        for patch in bp['boxes']:
            patch.set_alpha(0.7)

    # Shared formatting — spines and grid governed by stylesheet
    ax.set_title('Compare Models — CV MAE Folds', fontsize=14)
    ax.set_ylabel('MAE (dollars)', fontsize=12)
    ax.yaxis.set_major_formatter(
        m_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )

    plt.tight_layout()
    save_and_show_link(fig, f'plot_model_comparison_{get_current_timestamp()}.png')
    plt.close(fig)

    logger.debug("... FINISH")
    return fig