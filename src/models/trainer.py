"""
trainer.py — Model training and hyperparameter tuning module.

Design principles:
  - One generic `run_hyperparam_tuning` replaces model-specific functions.
    Model-specific logic lives in small `*_param_space` and `build_*` functions.
  - TPE sampler (Optuna default) replaces RandomSampler for smarter search.
  - CV mean + std both logged; objective is RMSE (same scale as SalePrice).
  - Logging dual-channel: structured file logs via project logger + rich
    console output for key notebook milestones (start, new best, summary).
  - RandomForestRegressor used (not Classifier) — this is a regression task.
  - All functions have docstrings; no commented-out print blocks.
"""

import os
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from rich.table import Table
from rich.panel import Panel
from rich import box
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost

from src.config.settings import RANDOM_STATE, PROJECT_NAME
from src.features.engineering import create_final_pipeline
from src.utils.logging import get_logger, console
from src.visualisation.plots import plot_residuals

logger = get_logger()

# ---------------------------------------------------------------------------
# Optuna verbosity — WARNING keeps useful progress messages during tuning
# while suppressing per-trial debug noise. Flip to ERROR for silent runs.
# ---------------------------------------------------------------------------
OPTUNA_LOG_LEVEL = optuna.logging.WARNING


# ===========================================================================
# Model-specific parameter spaces
# Each function takes an Optuna `trial` and returns a plain params dict.
# Add new model spaces here without touching the generic tuning function.
# ===========================================================================

def lasso_param_space(trial: optuna.Trial) -> dict:
    """Lasso regression hyperparameter search space."""
    return {
        'alpha':         trial.suggest_float('alpha', 1e-2, 10.0, log=True),
        'max_iter':      trial.suggest_int('max_iter', 50_000, 200_000),
        'tol':           trial.suggest_float('tol', 1e-4, 1e-2, log=True),
        'selection':     trial.suggest_categorical('selection', ['cyclic', 'random']),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }


def xgb_param_space(trial: optuna.Trial) -> dict:
    """XGBoost regressor hyperparameter search space."""
    return {
        'max_depth':        trial.suggest_int('max_depth', 2, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'gamma':            trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }


def rfr_param_space(trial: optuna.Trial) -> dict:
    """Random Forest regressor hyperparameter search space."""
    return {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
        'max_depth':         trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap':         trial.suggest_categorical('bootstrap', [True, False]),
    }


# ===========================================================================
# Model factory helpers
# Map a params dict → instantiated (unfitted) model.
# random_state fixed here ensures reproducibility of the final model.
# ===========================================================================

def build_lasso(params: dict) -> Lasso:
    """Instantiate a Lasso model from a params dict."""
    return Lasso(**params, random_state=RANDOM_STATE)


def build_xgb(params: dict) -> xgboost.XGBRegressor:
    """Instantiate an XGBoost regressor from a params dict."""
    return xgboost.XGBRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        enable_categorical=True,
    )


def build_rfr(params: dict) -> RandomForestRegressor:
    """Instantiate a Random Forest regressor from a params dict."""
    return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)


# ===========================================================================
# Optuna callbacks
# ===========================================================================

def champion_callback(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
    """
    Fires after each trial. Logs and displays a rich message whenever a new
    best score is found. Correctly handles the very first trial (no prior best).
    """
    prev_best    = study.user_attrs.get("best_rmse", None)
    current_rmse = frozen_trial.value  # objective returns RMSE

    if current_rmse is None:
        return  # pruned or failed trial — nothing to report

    if prev_best is None or current_rmse < prev_best:
        improvement_pct = (
            abs(prev_best - current_rmse) / prev_best * 100
            if prev_best is not None else None
        )
        study.set_user_attr("best_rmse", current_rmse)

        if improvement_pct is not None:
            msg = (
                f"Trial {frozen_trial.number}: RMSE {current_rmse:.4f} "
                f"({improvement_pct:.2f}% improvement over {prev_best:.4f})"
            )
        else:
            msg = f"Trial {frozen_trial.number}: initial best RMSE {current_rmse:.4f}"

        # Log to file via logger; display inline via rich console
        logger.info(msg)
        console.print(f"[bold green]▶ New best found[/bold green] — {msg}")


# ===========================================================================
# Cross-validation helper
# ===========================================================================

def _cross_val_rmse(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float]:
    """
    Run KFold cross-validation and return (mean_rmse, std_rmse).

    Parameters
    ----------
    model : fitted-ready sklearn-compatible estimator
    X_train : array-like of shape (n_samples, n_features)
        Accepts both DataFrame and ndarray.
    y_train : array-like of shape (n_samples,)
    n_splits : int
        Number of KFold splits. Default 5.

    Returns
    -------
    mean_rmse, std_rmse : floats
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        # Support both ndarray and DataFrame inputs transparently
        if hasattr(X_train, 'iloc'):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
        else:
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)

        if np.isnan(preds).any():
            raise ValueError("Model produced NaN predictions during CV")

        rmse_scores.append(np.sqrt(mean_squared_error(y_vl, preds)))

    return float(np.mean(rmse_scores)), float(np.std(rmse_scores))


# ===========================================================================
# Generic hyperparameter tuning function
# ===========================================================================

def run_hyperparam_tuning(
    model_name: str,
    param_space_fn,
    model_builder_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    pproc_pipeline,
    experiment_id: str,
    run_name: str,
    artefact_path: str,
    num_trials: int,
    n_cv_splits: int = 5,
) -> tuple[Any, Any]:
    """
    Generic Optuna + MLflow hyperparameter tuning loop.

    Model-specific behaviour is injected via `param_space_fn` and
    `model_builder_fn` — use the `tune_*` convenience wrappers below
    rather than calling this directly.

    Parameters
    ----------
    model_name : str
        Human-readable name used for logging/display (e.g. "XGBoost").
    param_space_fn : Callable[[optuna.Trial], dict]
        Suggests and returns a hyperparameter dict for one trial.
    model_builder_fn : Callable[[dict], estimator]
        Builds an unfitted model from a params dict.
    X_train, y_train : training features and labels.
        Accepted as DataFrame or ndarray (both handled in _cross_val_rmse).
    X_val, y_val : held-out validation set for final model evaluation.
    pproc_pipeline : sklearn ColumnTransformer
        Preprocessing pipeline; combined with the model in create_final_pipeline.
    experiment_id : str
        MLflow experiment ID — obtain via get_or_create_experiment().
    run_name : str
        Name for the parent MLflow run.
    artefact_path : str
        Relative path inside the MLflow run for the model artefact.
    num_trials : int
        Number of Optuna trials.
    n_cv_splits : int
        Number of KFold splits during cross-validation. Default 5.

    Returns
    -------
    tuple[optuna.Study, sklearn.Pipeline]
        (study, final_pipe) — study for Optuna inspection; final_pipe is the
        fitted preprocessor+model pipeline ready for predict() or feature importance.
    """
    logger.info(f"[{model_name}] Tuning started | trials={num_trials} | cv_splits={n_cv_splits}")
    console.rule(f"[bold blue]{model_name} — Tuning Started[/bold blue]")
    console.print(f"  {num_trials} Optuna trials · {n_cv_splits}-fold CV · objective: RMSE\n")

    # ------------------------------------------------------------------
    # Objective closure — captures pproc_pipeline directly from outer
    # scope to avoid the late-binding alias confusion.
    # ------------------------------------------------------------------
    def optuna_objective(trial: optuna.Trial) -> float:
        params     = param_space_fn(trial)
        model      = model_builder_fn(params)
        final_pipe = create_final_pipeline(pproc_pipeline, model)

        mean_rmse, std_rmse = _cross_val_rmse(final_pipe, X_train, y_train, n_cv_splits)

        # Log each trial as a nested MLflow run so they're queryable in the UI
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse_mean", mean_rmse)
            mlflow.log_metric("cv_rmse_std",  std_rmse)
            mlflow.log_metric("cv_mse_mean",  mean_rmse ** 2)

        logger.debug(
            f"[{model_name}] Trial {trial.number}: "
            f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}"
        )
        return mean_rmse  # Optuna minimises this

    # ------------------------------------------------------------------
    # Parent MLflow run — wraps the entire study
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(OPTUNA_LOG_LEVEL)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        mlflow.set_tags({
            "project":          PROJECT_NAME,
            "model_name":       model_name,
            "optimizer_engine": "optuna",
            "sampler":          "TPE",
        })

        # TPE sampler learns from past trials — far more efficient than random
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        best_rmse   = study.best_value
        best_params = study.best_params

        # Log study-level summary to the parent run
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_rmse",      best_rmse)
        mlflow.log_metric("best_cv_mse",       best_rmse ** 2)
        mlflow.log_metric("n_trials_completed", len(study.trials))

        logger.info(
            f"[{model_name}] Tuning complete | best_cv_rmse={best_rmse:.4f} | "
            f"best_params={best_params}"
        )

        # ------------------------------------------------------------------
        # Final model: retrain on full training set with best params,
        # then evaluate on the held-out validation set.
        # ------------------------------------------------------------------
        final_model = model_builder_fn(best_params)
        final_pipe  = create_final_pipeline(pproc_pipeline, final_model)
        final_pipe.fit(X_train, y_train)

        # Training metrics — logged alongside val metrics to spot overfitting
        y_train_pred = final_pipe.predict(X_train)
        train_rmse   = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2     = r2_score(y_train, y_train_pred)

        # Validation metrics — primary measure of generalisation
        y_val_pred = final_pipe.predict(X_val)
        val_rmse   = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2     = r2_score(y_val, y_val_pred)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2",   train_r2)
        mlflow.log_metric("val_rmse",   val_rmse)
        mlflow.log_metric("val_r2",     val_r2)

        logger.info(
            f"[{model_name}] Final model | "
            f"train_rmse={train_rmse:.4f} | train_r2={train_r2:.4f} | "
            f"val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}"
        )

        # Residuals plot persisted as an MLflow artefact
        residuals_fig = plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals_fig, "residuals_val.png")

        # Full pipeline (preprocessor + model) logged as an sklearn artefact
        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=(
                X_train.iloc[:3] if hasattr(X_train, 'iloc') else X_train[:3]
            ),
        )

        # Rich summary panel shown in notebook on completion
        console.rule(f"[bold green]{model_name} — Tuning Complete[/bold green]")
        summary = (
            f"Best CV RMSE : [bold green]{best_rmse:.4f}[/bold green]\n"
            f"Val RMSE     : [bold]{val_rmse:.4f}[/bold]\n"
            f"Val R²       : [bold]{val_r2:.4f}[/bold]"
        )
        console.print(Panel(summary, expand=False, border_style="green"))

    return study, final_pipe


# ===========================================================================
# Convenience wrappers — thin, named entry points for each model type.
# These keep call-sites in the notebook readable while all logic stays above.
# ===========================================================================

def tune_lasso(X_train, y_train, X_val, y_val, pproc_pipeline,
               experiment_id, run_name, artefact_path, num_trials) -> tuple:
    """Run Optuna + MLflow tuning for Lasso regression. Returns (study, final_pipe)."""
    return run_hyperparam_tuning(
        model_name="Lasso",
        param_space_fn=lasso_param_space,
        model_builder_fn=build_lasso,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        pproc_pipeline=pproc_pipeline,
        experiment_id=experiment_id,
        run_name=run_name,
        artefact_path=artefact_path,
        num_trials=num_trials,
    )


def tune_xgb(X_train, y_train, X_val, y_val, pproc_pipeline,
             experiment_id, run_name, artefact_path, num_trials) -> tuple:
    """Run Optuna + MLflow tuning for XGBoost regressor. Returns (study, final_pipe)."""
    return run_hyperparam_tuning(
        model_name="XGBoost",
        param_space_fn=xgb_param_space,
        model_builder_fn=build_xgb,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        pproc_pipeline=pproc_pipeline,
        experiment_id=experiment_id,
        run_name=run_name,
        artefact_path=artefact_path,
        num_trials=num_trials,
    )


def tune_rfr(X_train, y_train, X_val, y_val, pproc_pipeline,
             experiment_id, run_name, artefact_path, num_trials) -> tuple:
    """Run Optuna + MLflow tuning for Random Forest regressor. Returns (study, final_pipe)."""
    return run_hyperparam_tuning(
        model_name="RandomForestRegressor",
        param_space_fn=rfr_param_space,
        model_builder_fn=build_rfr,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        pproc_pipeline=pproc_pipeline,
        experiment_id=experiment_id,
        run_name=run_name,
        artefact_path=artefact_path,
        num_trials=num_trials,
    )


# ===========================================================================
# MLflow experiment helpers
# ===========================================================================

def get_or_create_experiment(experiment_name: str) -> str:
    """
    Return the experiment ID for `experiment_name`, creating it if absent.

    Parameters
    ----------
    experiment_name : str

    Returns
    -------
    str : MLflow experiment ID
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def set_mlflow_uri(uri_value: str) -> None:
    """
    Set the MLflow tracking URI — i.e. where run data is stored.

    Call before set_mlflow_experiment() and any tuning functions.

    Parameters
    ----------
    uri_value : str
        e.g. 'sqlite:///mlruns.db' for a local SQLite backend,
        or 'http://localhost:5000' for a remote MLflow server.
    """
    mlflow.set_tracking_uri(uri_value)


def set_mlflow_experiment(experiment_name: str) -> None:
    """
    Set the active MLflow experiment (the named folder within the tracking store).

    All subsequent runs will be logged under this experiment.

    Parameters
    ----------
    experiment_name : str
    """
    mlflow.set_experiment(experiment_name)


# ===========================================================================
# Model persistence
# ===========================================================================

def load_model(model_path: str):
    """
    Load a joblib-serialised model from disk.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the .pkl / .joblib file.

    Returns
    -------
    Loaded model object.

    Raises
    ------
    FileNotFoundError : if the path does not exist.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as exc:
        logger.error(f"Failed to load model from {model_path}: {exc}")
        raise


def save_model(model, model_path: str) -> None:
    """
    Persist a model to disk with joblib.

    Parameters
    ----------
    model : fitted estimator
    model_path : str
        Destination file path — directories are created if absent.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


# ===========================================================================
# Analysis, Comparison & Visualisation Utilities
#
# Logging strategy used throughout:
#   logger.info()  → persisted to log file AND rendered via RichHandler
#                    (use for scalar facts: scores, counts, param values)
#   console.*()    → rich-only structured output: tables, rules, panels
#                    (use for formatted display that has no log-file value)
# ===========================================================================

def get_runs_df(experiment_id: str) -> pd.DataFrame:
    """
    Fetch all parent-level MLflow runs for an experiment as a DataFrame.

    Filters out nested trial runs, returning only the model-level parent
    runs. Sorted by val_rmse ascending so the best model appears first.

    Parameters
    ----------
    experiment_id : str

    Returns
    -------
    pd.DataFrame with columns: model, cv_rmse, val_rmse, val_r2,
    train_rmse, train_r2, run_id. Empty DataFrame if no runs found.
    """
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",  # parent runs only
    )

    if runs_df.empty:
        logger.warning(f"No runs found for experiment_id={experiment_id}")
        return runs_df

    # Map verbose MLflow column names to short readable names
    col_map = {
        'tags.model_name':      'model',
        'metrics.best_cv_rmse': 'cv_rmse',
        'metrics.train_rmse':   'train_rmse',
        'metrics.val_rmse':     'val_rmse',
        'metrics.val_r2':       'val_r2',
        'metrics.train_r2':     'train_r2',
        'run_id':               'run_id',
    }
    available = {k: v for k, v in col_map.items() if k in runs_df.columns}
    result = runs_df[list(available.keys())].rename(columns=available)

    if 'val_rmse' in result.columns:
        result = result.sort_values('val_rmse').reset_index(drop=True)

    logger.info(f"Fetched {len(result)} parent runs from experiment_id={experiment_id}")
    return result


def print_study_summary(studies: dict) -> None:
    """
    Display a structured summary for one or more completed Optuna studies.

    Logs key scalar facts (best RMSE, trial count) via logger so they are
    captured in the log file. Renders the best-params breakdown as a rich
    table — structured layout that adds no value in a log file.

    Parameters
    ----------
    studies : dict[str, optuna.Study]
        Mapping of model name → completed study.
        e.g. {"XGBoost": study_xgb, "Lasso": study_lasso}
    """
    for model_name, study in studies.items():

        # Scalar facts → logger (file + console via RichHandler)
        logger.info(
            f"[{model_name}] best_cv_rmse={study.best_value:.4f} | "
            f"best_trial=#{study.best_trial.number} | "
            f"total_trials={len(study.trials)}"
        )

        # Section separator — console only, no log-file value
        console.rule(f"[bold blue]{model_name} — Study Summary[/bold blue]")

        # Key scalars as a compact panel
        summary = (
            f"Best CV RMSE : [bold green]{study.best_value:.4f}[/bold green]\n"
            f"Best trial # : [bold]#{study.best_trial.number}[/bold]\n"
            f"Total trials : [bold]{len(study.trials)}[/bold]"
        )
        console.print(Panel(summary, expand=False, border_style="blue"))

        # Best params table — one row per param
        table = Table(
            title="Best Parameters",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            title_style="bold",
            min_width=40,
        )
        table.add_column("Parameter",  style="dim")
        table.add_column("Best Value", justify="right")

        for param, value in study.best_params.items():
            logger.debug(f"  [{model_name}] {param} = {value}")
            table.add_row(param, str(value))

        console.print(table)


def print_trials_summary(study: optuna.Study, model_name: str, top_n: int = 10) -> None:
    """
    Display the top-N trials from a completed study ranked by CV RMSE.

    Each trial's rank and score is logged via logger for file persistence.
    The ranked table is rendered via console for structured display.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    model_name : str
        Used in the section header and log messages.
    top_n : int
        Number of top-performing trials to display. Default 10.
    """
    trials_df = study.trials_dataframe(attrs=('number', 'value', 'state'))
    completed = (
        trials_df[trials_df['state'] == 'COMPLETE']
        .nsmallest(top_n, 'value')
        .reset_index(drop=True)
    )

    if completed.empty:
        logger.warning(f"[{model_name}] No completed trials found in study")
        return

    logger.info(f"[{model_name}] Top-{top_n} trials by CV RMSE:")

    console.rule(f"[bold blue]{model_name} — Top {top_n} Trials[/bold blue]")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank",    justify="right", style="dim")
    table.add_column("Trial #", justify="right")
    table.add_column("CV RMSE", justify="right", style="bold green")

    for rank, (_, row) in enumerate(completed.iterrows(), start=1):
        trial_num = int(row['number'])
        rmse      = row['value']

        # Log each entry so rankings are captured in the log file
        logger.info(f"  [{model_name}] Rank {rank:>2} | Trial #{trial_num} | RMSE={rmse:.4f}")

        table.add_row(str(rank), str(trial_num), f"{rmse:.4f}")

    console.print(table)


def print_runs_comparison(experiment_id: str) -> None:
    """
    Display a cross-model comparison table fetched from MLflow.

    Pulls all parent-level runs for the experiment, sorts by val_rmse, and
    renders a rich table with an overfitting indicator (train_rmse - val_rmse).

    The winner (lowest val_rmse) is highlighted in bold green. The overfit
    gap is colour-coded: green within tolerance, red if gap exceeds 5000.

    Parameters
    ----------
    experiment_id : str
        MLflow experiment ID — obtain via get_or_create_experiment().
    """
    df = get_runs_df(experiment_id)

    if df.empty:
        logger.warning(f"No runs found for experiment_id={experiment_id}")
        console.print("[yellow]No runs found — have any tuning functions been run yet?[/yellow]")
        return

    logger.info(f"Model comparison | {len(df)} runs | experiment_id={experiment_id}")

    console.rule("[bold blue]Model Comparison — best val RMSE first[/bold blue]")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        caption="Overfit gap = train_rmse − val_rmse  ·  [green]green[/green] ≤ 5000  [red]red[/red] > 5000",
        caption_style="dim",
    )
    table.add_column("Rank",        justify="right", style="dim")
    table.add_column("Model",       justify="left")
    table.add_column("CV RMSE",     justify="right")
    table.add_column("Val RMSE",    justify="right")
    table.add_column("Val R²",      justify="right")
    table.add_column("Train RMSE",  justify="right")
    table.add_column("Overfit gap", justify="right")

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        model_name = row.get('model',      '?')
        cv_rmse    = row.get('cv_rmse',    float('nan'))
        val_rmse   = row.get('val_rmse',   float('nan'))
        val_r2     = row.get('val_r2',     float('nan'))
        train_rmse = row.get('train_rmse', float('nan'))
        gap        = train_rmse - val_rmse

        # Colour-code the gap — large positive gap signals overfitting
        gap_str = (
            f"[red]{gap:.1f}[/red]"
            if abs(gap) > 5000
            else f"[green]{gap:.1f}[/green]"
        )

        # Highlight the best model (rank 1) in bold green
        name_str = f"[bold]{model_name}[/bold]" if rank == 1 else model_name
        val_str  = f"[bold green]{val_rmse:.1f}[/bold green]" if rank == 1 else f"{val_rmse:.1f}"

        # Log each row so the comparison is captured in the log file
        logger.info(
            f"  Rank {rank} | {model_name} | "
            f"cv_rmse={cv_rmse:.1f} | val_rmse={val_rmse:.1f} | "
            f"val_r2={val_r2:.4f} | gap={gap:.1f}"
        )

        table.add_row(
            str(rank),
            name_str,
            f"{cv_rmse:.1f}",
            val_str,
            f"{val_r2:.4f}",
            f"{train_rmse:.1f}",
            gap_str,
        )

    console.print(table)
