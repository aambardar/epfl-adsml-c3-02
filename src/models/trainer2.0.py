"""
trainer2.0.py — Redesigned model training and hyperparameter tuning module.

Design principles vs trainer.py:
  - One generic `run_hyperparam_tuning` replaces four near-identical model-specific
    functions.  Model-specific logic lives in small `*_param_space` functions.
  - TPE sampler (Optuna default) replaces RandomSampler for smarter search.
  - CV mean + std both logged; objective is RMSE (same scale as SalePrice).
  - Logging dual-channel: structured file logs via project logger + styled
    IPython HTML display for key notebook milestones (start, new best, summary).
  - RandomForestRegressor used instead of the (incorrect) RandomForestClassifier.
  - No-op self-assignments, dead SQLite counter, and thin MLflow wrappers removed.
  - All functions have docstrings; no commented-out print blocks.
"""

import math
import os

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost

from src.config.settings import RANDOM_STATE, PROJECT_NAME
from src.features.engineering import create_final_pipeline
from src.utils.logging import get_logger
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
    return {
        'alpha':         trial.suggest_float('alpha', 1e-2, 10.0, log=True),
        'max_iter':      trial.suggest_int('max_iter', 50_000, 200_000),
        'tol':           trial.suggest_float('tol', 1e-4, 1e-2, log=True),
        'selection':     trial.suggest_categorical('selection', ['cyclic', 'random']),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
    }


def xgb_param_space(trial: optuna.Trial) -> dict:
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
    """Random Forest *Regressor* search space (regression task)."""
    return {
        'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
        'max_depth':        trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap':        trial.suggest_categorical('bootstrap', [True, False]),
    }


# ===========================================================================
# Model factory helpers
# Map a params dict → instantiated (unfitted) model.
# Keeping random_state here ensures reproducibility of the final model.
# ===========================================================================

def build_lasso(params: dict) -> Lasso:
    return Lasso(**params, random_state=RANDOM_STATE)


def build_xgb(params: dict) -> xgboost.XGBRegressor:
    return xgboost.XGBRegressor(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        enable_categorical=True,
    )


def build_rfr(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)


# ===========================================================================
# Notebook display helpers
# These emit styled HTML so key events stand out in the notebook cell output
# while the same information is also captured in the log file via logger.
# ===========================================================================

def _nb_display(html: str) -> None:
    """Emit an HTML block to the notebook output. No-op outside Jupyter."""
    try:
        display(HTML(html))
    except Exception:
        pass  # graceful degradation outside notebook environments


def _nb_info(title: str, body: str = "") -> None:
    """Blue info banner for general milestones."""
    _nb_display(
        f'<div style="border-left:4px solid #4a90d9;padding:6px 12px;'
        f'background:#eef5fb;margin:4px 0">'
        f'<b style="color:#1a5f9c">{title}</b>'
        f'{f"<br><span style=\'color:#333\'>{body}</span>" if body else ""}'
        f'</div>'
    )


def _nb_success(title: str, body: str = "") -> None:
    """Green banner for new best results."""
    _nb_display(
        f'<div style="border-left:4px solid #27ae60;padding:6px 12px;'
        f'background:#eafaf1;margin:4px 0">'
        f'<b style="color:#1e8449">{title}</b>'
        f'{f"<br><span style=\'color:#333\'>{body}</span>" if body else ""}'
        f'</div>'
    )


def _nb_summary_table(study: optuna.Study, model_name: str) -> None:
    """Render a compact summary table of the top-5 trials in the notebook."""
    trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    completed = trials_df[trials_df['state'] == 'COMPLETE'].nsmallest(5, 'value')
    if completed.empty:
        return
    # Build a minimal HTML table of top-5 trials
    rows = "".join(
        f"<tr><td>{int(r['number'])}</td><td>{r['value']:.4f}</td></tr>"
        for _, r in completed.iterrows()
    )
    _nb_display(
        f'<div style="margin:8px 0">'
        f'<b>{model_name} — top-5 trials (by CV RMSE)</b>'
        f'<table style="border-collapse:collapse;font-size:13px;margin-top:4px">'
        f'<tr style="background:#dce8f5"><th style="padding:3px 10px">Trial</th>'
        f'<th style="padding:3px 10px">CV RMSE</th></tr>'
        f'{rows}'
        f'</table></div>'
    )


# ===========================================================================
# Optuna callbacks
# ===========================================================================

def champion_callback(study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
    """
    Fires after each trial. Logs and displays a banner whenever a new best
    score is found.  Correctly handles the very first trial (no prior best).
    """
    prev_best = study.user_attrs.get("best_rmse", None)
    current_rmse = frozen_trial.value  # objective returns RMSE

    if current_rmse is None:
        return  # pruned or failed trial

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

        logger.info(msg)
        _nb_success("New best found", msg)


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
    y_train : array-like of shape (n_samples,)
    n_splits : int

    Returns
    -------
    mean_rmse, std_rmse : floats
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        # Support both ndarray and DataFrame inputs
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
) -> optuna.Study:
    """
    Generic Optuna + MLflow hyperparameter tuning loop.

    Replaces the four model-specific `run_hyperparam_tuning_*` functions in
    trainer.py with a single, model-agnostic implementation.  Model-specific
    behaviour is injected via `param_space_fn` and `model_builder_fn`.

    Parameters
    ----------
    model_name : str
        Human-readable name used for logging/display (e.g. "XGBoost").
    param_space_fn : Callable[[optuna.Trial], dict]
        Suggests and returns hyperparameter dict for one trial.
    model_builder_fn : Callable[[dict], estimator]
        Builds an unfitted model from a params dict.
    X_train, y_train : training features and labels
        Accepted as DataFrame/ndarray (both paths handled in _cross_val_rmse).
    X_val, y_val : held-out validation set for final model evaluation.
    pproc_pipeline : fitted sklearn ColumnTransformer
        Preprocessing pipeline; combined with the model inside create_final_pipeline.
    experiment_id : str
        MLflow experiment ID (obtain via get_or_create_experiment).
    run_name : str
        Name for the parent MLflow run.
    artefact_path : str
        Relative path inside the MLflow run where the model artefact is stored.
    num_trials : int
        Number of Optuna trials.
    n_cv_splits : int
        Number of KFold splits during cross-validation. Default 5.

    Returns
    -------
    optuna.Study
        Completed study; caller can inspect best_params, best_value, etc.
    """
    logger.info(f"[{model_name}] Hyperparameter tuning started | trials={num_trials} | cv_splits={n_cv_splits}")
    _nb_info(
        f"{model_name} — tuning started",
        f"{num_trials} Optuna trials · {n_cv_splits}-fold CV · objective: RMSE",
    )

    # ------------------------------------------------------------------
    # Objective closure — captures everything it needs from outer scope.
    # pproc_pipeline is referenced directly (not via a reassigned alias)
    # to avoid the closure-scope confusion in the original code.
    # ------------------------------------------------------------------
    def optuna_objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        model = model_builder_fn(params)
        final_pipe = create_final_pipeline(pproc_pipeline, model)

        mean_rmse, std_rmse = _cross_val_rmse(final_pipe, X_train, y_train, n_cv_splits)

        # Log each trial as a nested MLflow run so they're queryable in the UI
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse_mean", mean_rmse)
            mlflow.log_metric("cv_rmse_std", std_rmse)
            mlflow.log_metric("cv_mse_mean", mean_rmse ** 2)

        logger.debug(
            f"[{model_name}] Trial {trial.number}: "
            f"RMSE={mean_rmse:.4f} ± {std_rmse:.4f}"
        )
        return mean_rmse  # Optuna minimises this

    # ------------------------------------------------------------------
    # Parent MLflow run wraps the entire study
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(OPTUNA_LOG_LEVEL)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        mlflow.set_tags({
            "project": PROJECT_NAME,
            "model_name": model_name,
            "optimizer_engine": "optuna",
            "sampler": "TPE",
        })

        # TPE sampler: learns from past trials, far more efficient than Random
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        best_rmse = study.best_value
        best_params = study.best_params

        # Log study-level summary to the parent run
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_rmse", best_rmse)
        mlflow.log_metric("best_cv_mse", best_rmse ** 2)
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
        final_pipe = create_final_pipeline(pproc_pipeline, final_model)
        final_pipe.fit(X_train, y_train)

        # Training metrics (to spot overfitting)
        y_train_pred = final_pipe.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        # Validation metrics
        y_val_pred = final_pipe.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        logger.info(
            f"[{model_name}] Final model | "
            f"train_rmse={train_rmse:.4f} | train_r2={train_r2:.4f} | "
            f"val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}"
        )

        # Residuals plot logged as an MLflow artefact
        residuals_fig = plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals_fig, "residuals_val.png")

        # Log the full pipeline (preprocessor + model) as an sklearn model artefact
        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=(
                X_train.iloc[:3] if hasattr(X_train, 'iloc') else X_train[:3]
            ),
        )

        # Notebook summary
        _nb_success(
            f"{model_name} — tuning complete",
            f"Best CV RMSE: {best_rmse:.4f} &nbsp;|&nbsp; "
            f"Val RMSE: {val_rmse:.4f} &nbsp;|&nbsp; Val R²: {val_r2:.4f}",
        )
        _nb_summary_table(study, model_name)

    return study


# ===========================================================================
# Convenience wrappers — thin, named entry points for each model type.
# These keep call-sites in the notebook readable while all logic stays above.
# ===========================================================================

def tune_lasso(X_train, y_train, X_val, y_val, pproc_pipeline,
               experiment_id, run_name, artefact_path, num_trials) -> optuna.Study:
    """Run Optuna + MLflow tuning for Lasso regression."""
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
             experiment_id, run_name, artefact_path, num_trials) -> optuna.Study:
    """Run Optuna + MLflow tuning for XGBoost regressor."""
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
             experiment_id, run_name, artefact_path, num_trials) -> optuna.Study:
    """Run Optuna + MLflow tuning for Random Forest regressor."""
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
        Destination file path (directories are created if absent).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
