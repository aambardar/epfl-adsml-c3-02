"""
trainer.py — Model training and hyperparameter tuning module.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import xgboost

from src.config.settings import RANDOM_STATE, PROJECT_NAME, MLFLOW_TRACKING_URI
from src.features.engineering import create_final_pipeline
from src.utils.logging import get_logger
from src.visualisation.plots import plot_residuals, save_and_show_link

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
    Fires after each trial. Logs and prints a message whenever a new best
    score is found. Correctly handles the very first trial (no prior best).
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

        logger.info(msg)
        print(f"  >> New best -- {msg}")


# ===========================================================================
# Cross-validation helper
# ===========================================================================
def _cross_val_scores(
model,
X_train: pd.DataFrame | np.ndarray,
y_train: np.ndarray,
n_splits: int = 5,
log_target: bool = False,
features: list[str] | None = None,
) -> dict:
    """
    Run KFold cross-validation and return per-fold RMSE and MAE scores.
    Per-fold arrays are returned alongside mean/std so callers can compute
    any aggregation (median, IQR, box plot) without re-running CV.

    Parameters
    ----------
    model : sklearn-compatible estimator with fit/predict
    X_train : DataFrame or ndarray of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    n_splits : int
      Number of KFold splits. Default 5.
    log_target : bool
      If True, exp() is applied to predictions and actuals before
      computing MAE so it is always in dollars. RMSE is always returned
      in the training scale (log or raw). Default False.
    features : list[str] or None
      If provided, X_train is sliced to these columns before CV.
      Requires X_train to be a DataFrame. Default None (use all columns).

    Returns
    -------
    dict with keys:
      rmse_folds  — list of per-fold RMSE values (training scale)
      mae_folds   — list of per-fold MAE values (dollars)
      rmse_mean   — float
      rmse_std    — float
      mae_mean    — float
      mae_std     — float

    Raises
    ------
    ValueError : if model produces NaN predictions in any fold
    KeyError   : if any feature in `features` is missing from X_train
    """
    logger.debug("START ...")
    if features is not None:
      if not hasattr(X_train, '__getitem__'):
          raise TypeError("[_cross_val_scores] features slice requires a DataFrame X_train")
      missing = [f for f in features if f not in X_train.columns]
      if missing:
          raise KeyError(f"[_cross_val_scores] Features not found in X_train: {missing}")
      X_train = X_train[features]

    y_train = np.asarray(y_train)

    logger.info(
        f"[CV] Starting | n_splits={n_splits} | log_target={log_target} | "
        f"features={features} | X_shape={np.shape(X_train)}"
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmse_folds, mae_folds = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
      if hasattr(X_train, 'iloc'):
          X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
      else:
          X_tr, X_vl = X_train[train_idx], X_train[val_idx]

      y_tr, y_vl = y_train[train_idx], y_train[val_idx]

      model.fit(X_tr, y_tr)
      preds = model.predict(X_vl)

      if np.isnan(preds).any():
          raise ValueError(f"[_cross_val_scores] NaN predictions in fold {fold}")

      # RMSE always in training scale
      rmse_folds.append(np.sqrt(mean_squared_error(y_vl, preds)))

      # MAE always in dollars — back-transform if log scale
      preds_dollars   = np.exp(preds)  if log_target else preds
      actuals_dollars = np.exp(y_vl)   if log_target else y_vl
      mae_folds.append(mean_absolute_error(actuals_dollars, preds_dollars))

      logger.debug(
          f"[CV] Fold {fold}/{n_splits} | "
          f"RMSE={rmse_folds[-1]:.4f} | MAE=${mae_folds[-1]:,.0f}"
      )

    result = {
        "rmse_folds": rmse_folds,
        "mae_folds": mae_folds,
        "rmse_mean": float(np.mean(rmse_folds)),
        "rmse_std": float(np.std(rmse_folds)),
        "mae_mean": float(np.mean(mae_folds)),
        "mae_std": float(np.std(mae_folds)),
    }

    logger.info(
      f"[CV] Complete | "
      f"RMSE={result['rmse_mean']:.4f} ± {result['rmse_std']:.4f} | "
      f"MAE=${result['mae_mean']:,.0f} ± ${result['mae_std']:,.0f}"
    )
    logger.debug("... FINISH")
    return result

# ===========================================================================
# Baseline model
# ===========================================================================

def run_baseline(
  y_train: np.ndarray,
  y_val: np.ndarray,
  log_target: bool = False,
  n_splits: int = 5,
) -> dict:
    """
    Evaluate a mean-prediction baseline via cross-validation.

    The baseline predicts the training set mean for every sample — equivalent
    to sklearn's DummyRegressor(strategy='mean'). It requires no features and
    performs no learning, establishing the performance floor all models must beat.

    If the target was log-transformed before training, set log_target=True so
    predictions are back-transformed (exp) before computing MAE in dollars.
    RMSE is always reported in the same scale as y_train/y_val (log or raw).

    Parameters
    ----------
    y_train : array-like of shape (n_samples,)
      Training target values (raw SalePrice or log-transformed).
    y_val : array-like of shape (n_samples,)
      Held-out validation target values (same scale as y_train).
    log_target : bool
      If True, exp() is applied to predictions and actuals before computing
      MAE so it is always reported in dollars regardless of training scale.
      Default False.
    n_splits : int
      Number of KFold splits for cross-validated MAE/RMSE. Default 5.

    Returns
    -------
    dict with keys:
      cv_mae_mean, cv_mae_std   — cross-validated MAE in dollars
      cv_rmse_mean, cv_rmse_std — cross-validated RMSE (training scale)
      val_mae                   — MAE in dollars on held-out val set
      val_rmse                  — RMSE on held-out val set (training scale)
    """
    logger.debug("START ...")
    logger.info(f"[Baseline] Starting | log_target={log_target} | cv_splits={n_splits}")
    print(f"\n{'='*60}\n  Baseline -- Mean Predictor\n{'='*60}")

    y_train = np.asarray(y_train)
    y_val   = np.asarray(y_val)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_mae, fold_rmse = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(y_train), start=1):
      y_tr, y_vl = y_train[train_idx], y_train[val_idx]

      # Baseline prediction: mean of the fold's training split
      fold_mean = np.mean(y_tr)
      # Using full_like to create a new array with the same shape as y_val, but filled entirely with a constant value.
      preds     = np.full_like(y_vl, fill_value=fold_mean, dtype=float)

      # Back-transform to dollars before scoring if target is log-scale
      preds_dollars = np.exp(preds) if log_target else preds
      actuals_dollars = np.exp(y_vl) if log_target else y_vl

      fold_mae.append(mean_absolute_error(actuals_dollars, preds_dollars))  # MAE in dollars
      fold_rmse.append(np.sqrt(mean_squared_error(y_vl, preds)))  # RMSE in training scale

      logger.debug(
          f"[Baseline] Fold {fold} | MAE=${fold_mae[-1]:,.0f} | RMSE={fold_rmse[-1]:.4f}"
      )

    cv_mae_mean  = float(np.mean(fold_mae))
    cv_mae_std   = float(np.std(fold_mae))
    cv_rmse_mean = float(np.mean(fold_rmse))
    cv_rmse_std  = float(np.std(fold_rmse))

    # Validation set evaluation
    val_mean          = np.mean(y_train)
    val_preds         = np.full_like(y_val, fill_value=val_mean, dtype=float)
    val_preds_dollars   = np.exp(val_preds) if log_target else val_preds
    val_actuals_dollars = np.exp(y_val)     if log_target else y_val

    val_mae  = mean_absolute_error(val_actuals_dollars, val_preds_dollars)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    logger.info(
      f"[Baseline] CV MAE=${cv_mae_mean:,.0f} ± ${cv_mae_std:,.0f} | "
      f"CV RMSE={cv_rmse_mean:.4f} ± {cv_rmse_std:.4f} | "
      f"Val MAE=${val_mae:,.0f} | Val RMSE={val_rmse:.4f}"
    )

    summary = (
      f"  Strategy     : predict mean of training set\n"
      f"  Log target   : {log_target}\n\n"
      f"  CV  MAE      : ${cv_mae_mean:>12,.0f} +/- ${cv_mae_std:,.0f}\n"
      f"  CV  RMSE     : {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f}\n\n"
      f"  Val MAE      : ${val_mae:>12,.0f}\n"
      f"  Val RMSE     : {val_rmse:.4f}"
    )
    print(f"\n  Baseline Results\n  {'-'*40}\n{summary}\n")

    logger.debug("... FINISH")
    return {
      "cv_mae_folds": fold_mae,
      "cv_mae_mean":  cv_mae_mean,
      "cv_mae_std":   cv_mae_std,
      "cv_rmse_mean": cv_rmse_mean,
      "cv_rmse_std":  cv_rmse_std,
      "val_mae":      val_mae,
      "val_rmse":     val_rmse,
    }

# ===========================================================================
# SelectKBest feature-selection model
# ===========================================================================
def run_simple_model(
X_train: pd.DataFrame,
y_train: np.ndarray,
X_val: pd.DataFrame,
y_val: np.ndarray,
k: int = 2,
log_target: bool = False,
n_splits: int = 5,
) -> dict:
    """
    Evaluate a LinearRegression model trained on the top-k MI-scored features.

    SelectKBest (mutual_info_regression scorer) ranks each feature by its
    mutual information with the target — capturing both linear and non-linear
    associations without assuming any functional form. This makes it more
    suitable than f_regression for this dataset, which has non-linearities
    (e.g. OverallQual) and mixed continuous/ordinal feature types.

    The selector is fit on X_train only; X_val is transformed using the same
    fitted selector to prevent data leakage. CV is then run on X_train_sel
    so each fold sees only the k selected columns.

    Parameters
    ----------
    X_train : DataFrame of shape (n_train, n_features)
      Training feature matrix.
    y_train : array-like of shape (n_train,)
      Training target values (raw SalePrice or log-transformed).
    X_val : DataFrame of shape (n_val, n_features)
      Held-out validation feature matrix (same columns as X_train).
    y_val : array-like of shape (n_val,)
      Held-out validation target values (same scale as y_train).
    k : int
      Number of top features to retain. Default 2.
    log_target : bool
      If True, exp() is applied before computing MAE so it is always
      reported in dollars. RMSE remains in training scale. Default False.
    n_splits : int
      Number of KFold splits for cross-validation. Default 5.

    Returns
    -------
    dict with keys:
    selected_features             — list[str] of the k retained column names
    cv_mae_mean, cv_mae_std       — cross-validated MAE in dollars
    cv_rmse_mean, cv_rmse_std     — cross-validated RMSE (training scale)
    val_mae                       — MAE in dollars on held-out val set
    val_rmse                      — RMSE on held-out val set (training scale)
    """
    logger.debug("START ...")
    logger.info(
      f"[SelectKBest] Starting | k={k} | log_target={log_target} | cv_splits={n_splits}"
    )
    print(f"\n{'='*60}\n  SelectKBest -- Mutual Information + LinearRegression\n{'='*60}")

    y_train = np.asarray(y_train)
    y_val   = np.asarray(y_val)

    # Feature selection (fit on train only to avoid leakage)
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_regression(X, y, random_state=RANDOM_STATE),
        k=k,
    )

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel   = selector.transform(X_val)

    # Recover column names for interpretability
    selected_features = list(X_train.columns[selector.get_support()])
    logger.info(f"[SelectKBest] Selected features: {selected_features}")

    # Choosing simple LR
    model  = LinearRegression()
    cv_res = _cross_val_scores(
      model, X_train_sel, y_train,
      n_splits=n_splits, log_target=log_target,
    )

    # Held-out validation evaluation
    model.fit(X_train_sel, y_train)
    val_preds = model.predict(X_val_sel)

    val_preds_dollars   = np.exp(val_preds) if log_target else val_preds
    val_actuals_dollars = np.exp(y_val)     if log_target else y_val

    val_mae  = mean_absolute_error(val_actuals_dollars, val_preds_dollars)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    logger.info(
      f"[SelectKBest] CV MAE=${cv_res['mae_mean']:,.0f} ± ${cv_res['mae_std']:,.0f} | "
      f"CV RMSE={cv_res['rmse_mean']:.4f} ± {cv_res['rmse_std']:.4f} | "
      f"Val MAE=${val_mae:,.0f} | Val RMSE={val_rmse:.4f}"
    )

    summary = (
      f"  Strategy      : SelectKBest (mutual_info_regression) + LinearRegression\n"
      f"  Features kept : {k} / {X_train.shape[1]}\n"
      f"  Log target    : {log_target}\n\n"
      f"  CV  MAE       : ${cv_res['mae_mean']:>12,.0f} +/- ${cv_res['mae_std']:,.0f}\n"
      f"  CV  RMSE      : {cv_res['rmse_mean']:.4f} +/- {cv_res['rmse_std']:.4f}\n\n"
      f"  Val MAE       : ${val_mae:>12,.0f}\n"
      f"  Val RMSE      : {val_rmse:.4f}"
    )
    print(f"\n  SelectKBest Results\n  {'-'*40}\n{summary}\n")

    logger.debug("... FINISH")
    return {
      "selected_features": selected_features,
      "cv_mae_folds":      cv_res["mae_folds"],
      "cv_mae_mean":       cv_res["mae_mean"],
      "cv_mae_std":        cv_res["mae_std"],
      "cv_rmse_mean":      cv_res["rmse_mean"],
      "cv_rmse_std":       cv_res["rmse_std"],
      "val_mae":           val_mae,
      "val_rmse":          val_rmse,
    }

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
    log_target: bool = False,
) -> tuple[Any, Any, Any]:
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
        Accepted as DataFrame or ndarray.
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
    tuple[optuna.Study, sklearn.Pipeline, dict]
        (study, final_pipe, metrics) — study for Optuna inspection; final_pipe is the
        fitted preprocessor+model pipeline ready for predict() or feature importance; and metrics is a dict
    """
    logger.debug("START ...")
    logger.info(f"[{model_name}] Tuning started | trials={num_trials} | cv_splits={n_cv_splits}")
    print(f"\n{'='*60}\n  {model_name} -- Tuning Started\n{'='*60}")
    print(f"  {num_trials} Optuna trials | {n_cv_splits}-fold CV | objective: RMSE\n")

    # ------------------------------------------------------------------
    # Objective closure — captures pproc_pipeline directly from outer
    # scope to avoid the late-binding alias confusion.
    # ------------------------------------------------------------------
    def optuna_objective(trial: optuna.Trial) -> float:
        params     = param_space_fn(trial)
        model      = model_builder_fn(params)
        final_pipe = create_final_pipeline(pproc_pipeline, model)

        cv = _cross_val_scores(final_pipe, X_train, y_train, n_cv_splits, log_target=log_target)
        trial.set_user_attr("cv_rmse_std", cv["rmse_std"])
        trial.set_user_attr("cv_mae_mean", cv["mae_mean"])
        trial.set_user_attr("cv_mae_std",  cv["mae_std"])
        trial.set_user_attr("cv_mae_folds", cv["mae_folds"])

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse_mean", cv["rmse_mean"])
            mlflow.log_metric("cv_rmse_std", cv["rmse_std"])
            mlflow.log_metric("cv_mse_mean", cv["rmse_mean"] ** 2)
            mlflow.log_metric("cv_mae_mean", cv["mae_mean"])
            mlflow.log_metric("cv_mae_std", cv["mae_std"])

        logger.debug(
            f"[{model_name}] Trial {trial.number}: "
            f"RMSE={cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f} | "
            f"MAE=${cv['mae_mean']:,.0f} ± ${cv['mae_std']:,.0f}"
        )
        return cv["rmse_mean"]  # Optuna minimises RMSE

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

        best_trial = study.best_trial
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

        # Validation metrics — the primary measure of generalisation
        y_val_pred = final_pipe.predict(X_val)
        val_preds_dollars = np.exp(y_val_pred) if log_target else y_val_pred
        val_actuals_dollars = np.exp(y_val) if log_target else y_val

        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))  # training scale
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(val_actuals_dollars, val_preds_dollars)  # dollars

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2",   train_r2)
        mlflow.log_metric("val_rmse",   val_rmse)
        mlflow.log_metric("val_r2",     val_r2)
        mlflow.log_metric("val_mae",    val_mae)

        logger.info(
            f"[{model_name}] Final model | "
            f"train_rmse={train_rmse:.4f} | train_r2={train_r2:.4f} | "
            f"val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f} | val_mae=${val_mae:,.0f}"
        )

        # Residuals plot persisted as an MLflow artefact
        residuals_fig = plot_residuals(y_val_pred, y_val)
        residuals_fig_name = f"{model_name}_{experiment_id}_residuals.png"
        mlflow.log_figure(residuals_fig, residuals_fig_name)
        save_and_show_link(residuals_fig, residuals_fig_name, show_link=False)

        # Full pipeline (preprocessor + model) logged as a sklearn artefact
        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=(
                X_train.iloc[:3] if hasattr(X_train, 'iloc') else X_train[:3]
            ),
        )

        print(f"\n{'='*60}\n  {model_name} -- Tuning Complete\n{'='*60}")
        summary = (
            f"  Best CV RMSE : {best_rmse:.4f}\n"
            f"  Val RMSE     : {val_rmse:.4f}\n"
            f"  Val R2       : {val_r2:.4f}\n"
            f"  Val MAE      : ${val_mae:>12,.0f}"
        )
        print(summary)
        print()

        metrics = {
            "cv_rmse_mean": best_rmse,
            "cv_rmse_std": best_trial.user_attrs["cv_rmse_std"],
            "cv_mae_mean": best_trial.user_attrs["cv_mae_mean"],
            "cv_mae_std": best_trial.user_attrs["cv_mae_std"],
            "cv_mae_folds": best_trial.user_attrs["cv_mae_folds"],
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "best_params": best_params,
            "n_trials": len(study.trials),
        }

    logger.debug("... FINISH")
    return study, final_pipe, metrics


# ===========================================================================
# Convenience wrappers — thin, named entry points for each model type.
# These keep call-sites in the notebook readable while all logic stays above.
# ===========================================================================

def tune_lasso(X_train, y_train, X_val, y_val, pproc_pipeline,
               experiment_id, run_name, artefact_path, num_trials, log_target: bool = False) -> tuple:
    """Run Optuna + MLflow tuning for Lasso regression. Returns (study, final_pipe, metrics)."""
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
        log_target=log_target,
    )


def tune_xgb(X_train, y_train, X_val, y_val, pproc_pipeline,
             experiment_id, run_name, artefact_path, num_trials, log_target: bool = False) -> tuple:
    """Run Optuna + MLflow tuning for XGBoost regressor. Returns (study, final_pipe, metrics)."""
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
        log_target=log_target,
    )


def tune_rfr(X_train, y_train, X_val, y_val, pproc_pipeline,
             experiment_id, run_name, artefact_path, num_trials, log_target: bool = False) -> tuple:
    """Run Optuna + MLflow tuning for Random Forest regressor. Returns (study, final_pipe, metrics)."""
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
        log_target=log_target,
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


def set_mlflow_uri(uri_value: str = MLFLOW_TRACKING_URI) -> None:
    """
    Set the MLflow tracking URI — i.e. where run data is stored.

    Call before set_mlflow_experiment() and any tuning functions.

    Parameters
    ----------
    uri_value : str
        e.g. 'sqlite:///mlruns.db' for a local SQLite backend,
        or 'http://localhost:5000' for a remote MLflow server.
    """
    logger.debug("START ...")
    mlflow.set_tracking_uri(uri_value)
    logger.debug("... FINISH")


def set_mlflow_experiment(experiment_name: str) -> None:
    """
    Set the active MLflow experiment (the named folder within the tracking store).

    All subsequent runs will be logged under this experiment.

    Parameters
    ----------
    experiment_name : str
    """
    logger.debug("START ...")
    mlflow.set_experiment(experiment_name)
    logger.debug("... FINISH")


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
    logger.debug("START ...")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        logger.debug("... FINISH")
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
    logger.debug("START ...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    logger.debug("... FINISH")


# ===========================================================================
# Analysis, Comparison & Visualisation Utilities
#
# Logging strategy used throughout:
#   logger.info()  -> persisted to log file AND rendered via console handler
#                     (use for scalar facts: scores, counts, param values)
#   print()        -> structured display output: tables, separators, panels
#                     (use for formatted display that has no log-file value)
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
    logger.debug("START ...")
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId = ''",  # parent runs only
    )

    if runs_df.empty:
        logger.warning(f"No runs found for experiment_id={experiment_id}")
        logger.debug("... FINISH")
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
    logger.debug("... FINISH")
    return result


def print_study_summary(studies: dict) -> None:
    """
    Display a structured summary for one or more completed Optuna studies.

    Logs key scalar facts (best RMSE, trial count) via logger so they are
    captured in the log file. Prints the best-params breakdown as a plain
    fixed-width table.

    Parameters
    ----------
    studies : dict[str, optuna.Study]
        Mapping of model name to completed study.
        e.g. {"XGBoost": study_xgb, "Lasso": study_lasso}
    """
    logger.debug("START ...")
    for model_name, study in studies.items():

        logger.info(
            f"[{model_name}] best_cv_rmse={study.best_value:.4f} | "
            f"best_trial=#{study.best_trial.number} | "
            f"total_trials={len(study.trials)}"
        )

        print(f"\n{'='*60}\n  {model_name} -- Study Summary\n{'='*60}")
        summary = (
            f"  Best CV RMSE : {study.best_value:.4f}\n"
            f"  Best trial # : #{study.best_trial.number}\n"
            f"  Total trials : {len(study.trials)}"
        )
        print(summary)

        print(f"\n  {'Parameter':<30}  {'Best Value':>15}")
        print(f"  {'-'*47}")
        for param, value in study.best_params.items():
            logger.debug(f"  [{model_name}] {param} = {value}")
            print(f"  {param:<30}  {str(value):>15}")
        print()
    logger.debug("... FINISH")


def print_trials_summary(study: optuna.Study, model_name: str, top_n: int = 10) -> None:
    """
    Display the top-N trials from a completed study ranked by CV RMSE.

    Each trial's rank and score is logged via logger for file persistence
    and printed as a plain fixed-width table.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    model_name : str
        Used in the section header and log messages.
    top_n : int
        Number of top-performing trials to display. Default 10.
    """
    logger.debug("START ...")
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

    print(f"\n{'='*60}\n  {model_name} -- Top {top_n} Trials\n{'='*60}")
    print(f"  {'Rank':>4}  {'Trial #':>7}  {'CV RMSE':>10}")
    print(f"  {'-'*25}")
    for rank, (_, row) in enumerate(completed.iterrows(), start=1):
        trial_num = int(row['number'])
        rmse      = row['value']
        logger.info(f"  [{model_name}] Rank {rank:>2} | Trial #{trial_num} | RMSE={rmse:.4f}")
        print(f"  {rank:>4}  {trial_num:>7}  {rmse:.4f}")
    print()
    logger.debug("... FINISH")


def print_runs_comparison(experiment_id: str) -> None:
    """
    Display a cross-model comparison table fetched from MLflow.

    Pulls all parent-level runs for the experiment, sorts by val_rmse, and
    prints a plain fixed-width table with an overfitting indicator
    (train_rmse - val_rmse). The best model (rank 1) is marked with an asterisk.
    Rows where the overfit gap exceeds 5000 are flagged with HIGH.

    Parameters
    ----------
    experiment_id : str
        MLflow experiment ID — obtain via get_or_create_experiment().
    """
    logger.debug("START ...")
    df = get_runs_df(experiment_id)

    if df.empty:
        logger.warning(f"No runs found for experiment_id={experiment_id}")
        print("  No runs found -- have any tuning functions been run yet?")
        return

    logger.info(f"Model comparison | {len(df)} runs | experiment_id={experiment_id}")

    print(f"\n{'='*60}\n  Model Comparison -- best val RMSE first\n{'='*60}")
    print(f"  Note: overfit gap = train_rmse - val_rmse  (* = best  HIGH = gap > 5000)\n")
    print(f"  {'':1}  {'Rank':>4}  {'Model':<20}  {'CV RMSE':>8}  {'Val RMSE':>9}  {'Val R2':>6}  {'Train RMSE':>10}  {'Gap':>8}")
    print(f"  {'-'*75}")

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        model_name = row.get('model',      '?')
        cv_rmse    = row.get('cv_rmse',    float('nan'))
        val_rmse   = row.get('val_rmse',   float('nan'))
        val_r2     = row.get('val_r2',     float('nan'))
        train_rmse = row.get('train_rmse', float('nan'))
        gap        = train_rmse - val_rmse
        gap_flag   = " HIGH" if abs(gap) > 5000 else "     "
        winner     = "*" if rank == 1 else " "

        logger.info(
            f"  Rank {rank} | {model_name} | "
            f"cv_rmse={cv_rmse:.1f} | val_rmse={val_rmse:.1f} | "
            f"val_r2={val_r2:.4f} | gap={gap:.1f}"
        )

        print(
            f"  {winner} {rank:>4}  {model_name:<20}  "
            f"{cv_rmse:>8.1f}  {val_rmse:>9.1f}  "
            f"{val_r2:>6.4f}  {train_rmse:>10.1f}  {gap:>8.1f}{gap_flag}"
        )

    print()
    logger.debug("... FINISH")