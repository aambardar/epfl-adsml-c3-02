import numpy as np
import math
import os
import joblib
import optuna

from proj_configs import RANDOM_STATE, PROJECT_NAME, PATH_OUT_MODELS
import mlflow
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import proj_utils
import proj_utils_plots
import proj_utils_feat_engg

from proj_utils_logging import get_logger
logger = get_logger()

LOG_OPTUNA_RUN_LEVEL= optuna.logging.ERROR

def create_metrics_snapshot():
    logger.debug("START ...")
    model_metrics = {
        'iteration': [],
        'model_name': [],
        'model_type': [],
        'train_mse': [],
        'val_mse': [],
        'test_mse': [],
        'train_r2': [],
        'val_r2': [],
        'test_r2': [],
    }
    logger.debug("... FINISH")
    return model_metrics

def update_metrics_snapshot(model_metrics, model_name, model_type, train_mse, val_mse, test_mse, train_r2, val_r2, test_r2):
    logger.debug("START ...")
    model_metrics['iteration'].append(len(model_metrics['iteration']) + 1)
    model_metrics['model_name'].append(model_name)
    model_metrics['model_type'].append(model_type)
    model_metrics['train_mse'].append(train_mse)
    model_metrics['val_mse'].append(val_mse)
    model_metrics['test_mse'].append(test_mse)
    model_metrics['train_r2'].append(train_r2)
    model_metrics['val_r2'].append(val_r2)
    model_metrics['test_r2'].append(test_r2)
    logger.debug("... FINISH")

def get_next_run_counter():
    """
    Fetches the last run counter from SQLite, increments it, and returns the new value.
    Creates the table if it doesn't exist.

    Returns:
        int: The incremented counter value
    """

    @contextmanager
    def get_db_connection():
        conn = sqlite3.connect('run_counter.db')
        try:
            yield conn
        finally:
            conn.close()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS run_counter
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY,
                           counter
                           INTEGER
                           NOT
                           NULL
                       )
                       ''')

        # Get the current counter value
        cursor.execute('SELECT counter FROM run_counter WHERE id = 1')
        result = cursor.fetchone()

        if result is None:
            # Initialize counter if no record exists
            new_counter = 1
            cursor.execute('INSERT INTO run_counter (id, counter) VALUES (1, ?)', (new_counter,))
        else:
            # Increment existing counter
            new_counter = result[0] + 1
            cursor.execute('UPDATE run_counter SET counter = ? WHERE id = 1', (new_counter,))

        conn.commit()
        return new_counter

def set_mlflow_uri(uri_value):
    mlflow.set_tracking_uri(uri_value)

def set_mlflow_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def save_features(filename, base_dir_path, features):
    logger.debug("START ...")
    proj_utils.save_file('feature', filename, base_dir_path, features)
    logger.debug("... FINISH")

def save_model(filename, base_dir_path, model):
    logger.debug("START ...")
    proj_utils.save_file('model', filename, base_dir_path, model)
    logger.debug("... FINISH")

def save_hyperparams(filename, base_dir_path, hyperparams):
    logger.debug("START ...")
    proj_utils.save_file('hyperparams', filename, base_dir_path, hyperparams)
    logger.debug("... FINISH")

def champion_callback(study, frozen_trial):
    logger.debug("START ...")
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            logger.info(f"=" * 50)
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
            logger.info(f"=" * 50)
        else:
            logger.info(f"=" * 50)
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"no improvement"
            )
            logger.info(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
            logger.info(f"=" * 50)
    logger.debug("... FINISH")

def run_hyperparam_tuning_lasso(X_train, y_train, X_val, y_val, pproc_pipeline, experiment_id, run_name, artefact_path, num_trials):
    def optuna_objective(trial):
        with mlflow.start_run(nested=True):
            # print(f'Trial: {proj_utils_plots.beautify(trial)}')
            # print(f'Trial number: {proj_utils_plots.beautify(trial.number)}')

            params_lasso = {
                'alpha': trial.suggest_float('alpha', 1e-2, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 50000, 200000),
                'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])
            }

            model_lasso = Lasso(**params_lasso)
            final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model_lasso)

            # print(f'Trial {proj_utils_plots.beautify(str(trial.number))} Scoring Starts...')
            cv_split_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # Perform cross-validation
            mse_scores = []
            for train_idx, val_idx in cv_split_kf.split(X_train, y_train):
                X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # Fit the model and make predictions
                final_pipe.fit(X_tr, y_tr)
                y_vl_pred = final_pipe.predict(X_vl)
                # Calculate MSE
                mse = mean_squared_error(y_vl, y_vl_pred)
                mse_scores.append(mse)

            score = np.mean(mse_scores)
            # Log to MLflow
            mlflow.log_params(params_lasso)
            mlflow.log_metric("mse", score)
            mlflow.log_metric("rmse", math.sqrt(score))

            # Print the results
            # print("MSE scores:", mse_scores)
            # print("Mean MSE score:", score)

        return score

    optuna.logging.set_verbosity(LOG_OPTUNA_RUN_LEVEL)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # creation of Optuna study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
        X_train = X_train
        y_train = y_train
        pproc_pipe = pproc_pipeline
        # optimise the study
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # print(
        #     f"Best trial: {proj_utils_plots.beautify(study.best_trial)} with value: {study.best_value}"
        # )

        # Log tags
        mlflow.set_tags(
            tags={
                "project": PROJECT_NAME,
                "optimizer_engine": "optuna"
            }
        )

        model = Lasso(**study.best_params)
        final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model)
        final_pipe.fit(X_train, y_train)
        y_val_pred = final_pipe.predict(X_val)
        residuals = proj_utils_plots.plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals, "residuals.png")

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_val_pred)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        artefact_path = artefact_path

        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=X_train.iloc[:3]
        )

    return study

def old_run_hyperparam_tuning_xgb(X_train, y_train, X_val, y_val, pproc_pipeline, experiment_id, run_name, artefact_path, num_trials):
    def optuna_objective(trial):
        with mlflow.start_run(nested=True):
            # print(f'Trial: {proj_utils_plots.beautify(trial)}')
            # print(f'Trial number: {proj_utils_plots.beautify(trial.number)}')

            params_xgb = {
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 10),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }

            model_xgb = xgboost.XGBRegressor(**params_xgb, n_jobs=-1, enable_categorical=True)
            final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model_xgb)

            # Add validation checks
            # if X_train.isnull().any().any() or y_train.isnull().any():
            #     raise ValueError("Training data contains NaN values")

            # print(f'Trial {proj_utils_plots.beautify(str(trial.number))} Scoring Starts...')
            cv_split_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # Perform cross-validation
            mse_scores = []
            for train_idx, val_idx in cv_split_kf.split(X_train, y_train):
                X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # Fit the model and make predictions
                final_pipe.fit(X_tr, y_tr)
                y_vl_pred = final_pipe.predict(X_vl)

                # Validate predictions
                if np.isnan(y_vl_pred).any():
                    raise ValueError("Model generated NaN predictions")

                # Calculate MSE
                mse = mean_squared_error(y_vl, y_vl_pred)
                mse_scores.append(mse)

            score = np.mean(mse_scores)
            # Log to MLflow
            mlflow.log_params(params_xgb)
            mlflow.log_metric("mse", score)
            mlflow.log_metric("rmse", math.sqrt(score))

            # Print the results
            # print("MSE scores:", mse_scores)
            # print("Mean MSE score:", score)

        return score

    optuna.logging.set_verbosity(LOG_OPTUNA_RUN_LEVEL)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # creation of Optuna study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
        X_train = X_train
        y_train = y_train
        pproc_pipe = pproc_pipeline
        # optimise the study
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # print(
        #     f"Best trial: {proj_utils_plots.beautify(study.best_trial)} with value: {study.best_value}"
        # )

        # Log tags
        mlflow.set_tags(
            tags={
                "project": PROJECT_NAME,
                "optimizer_engine": "optuna"
            }
        )

        model = xgboost.XGBRegressor(**study.best_params)
        final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model)
        final_pipe.fit(X_train, y_train)
        y_val_pred = final_pipe.predict(X_val)

        # Log the residual plot
        residuals = proj_utils_plots.plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals, "residuals.png")

        # Log the feature importance's plot
        # importances = proj_utils_plots.plot_feature_importance(model, booster=study.best_params.get("booster"))
        # mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_val_pred)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        artefact_path = artefact_path

        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=X_train.iloc[:3]
        )

        mlflow.xgboost.log_model(
            xgb_model=model,
            input_example=X_train.iloc[:3]
        )

    return study

def run_hyperparam_tuning_rfc(X_train, y_train, X_val, y_val, pproc_pipeline, experiment_id, run_name, artefact_path, num_trials):
    def optuna_objective(trial):
        with mlflow.start_run(nested=True):
            # print(f'Trial: {proj_utils_plots.beautify(trial)}')
            # print(f'Trial number: {proj_utils_plots.beautify(trial.number)}')

            params_rfc = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }

            model_rfc = RandomForestClassifier(**params_rfc)
            final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model_rfc)

            # print(f'Trial {proj_utils_plots.beautify(str(trial.number))} Scoring Starts...')
            cv_split_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # Perform cross-validation
            mse_scores = []
            for train_idx, val_idx in cv_split_kf.split(X_train, y_train):
                X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
                # Fit the model and make predictions
                final_pipe.fit(X_tr, y_tr)
                y_vl_pred = final_pipe.predict(X_vl)
                # Calculate MSE
                mse = mean_squared_error(y_vl, y_vl_pred)
                mse_scores.append(mse)

            score = np.mean(mse_scores)
            # Log to MLflow
            mlflow.log_params(params_rfc)
            mlflow.log_metric("mse", score)
            mlflow.log_metric("rmse", math.sqrt(score))

            # Print the results
            # print("MSE scores:", mse_scores)
            # print("Mean MSE score:", score)

        return score

    optuna.logging.set_verbosity(LOG_OPTUNA_RUN_LEVEL)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # creation of Optuna study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
        X_train = X_train
        y_train = y_train
        pproc_pipe = pproc_pipeline
        # optimise the study
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # print(
        #     f"Best trial: {proj_utils_plots.beautify(study.best_trial)} with value: {study.best_value}"
        # )

        # Log tags
        mlflow.set_tags(
            tags={
                "project": PROJECT_NAME,
                "optimizer_engine": "optuna"
            }
        )

        model = RandomForestClassifier(**study.best_params)
        final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model)
        final_pipe.fit(X_train, y_train)
        y_val_pred = final_pipe.predict(X_val)
        residuals = proj_utils_plots.plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals, "residuals.png")

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_val_pred)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        artefact_path = artefact_path

        mlflow.sklearn.log_model(
            sk_model=final_pipe,
            artifact_path=artefact_path,
            input_example=X_train.iloc[:3]
        )

    return study

def run_hyperparam_tuning_xgb(X_train_features, y_train, X_val_features, y_val, experiment_id, run_name, artefact_path, num_trials):
    def optuna_objective(trial):
        with mlflow.start_run(nested=True):
            # print(f'Trial: {proj_utils_plots.beautify(trial)}')
            # print(f'Trial number: {proj_utils_plots.beautify(trial.number)}')

            params_xgb = {
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 2),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9)
                # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                # 'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                # 'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }

            model_xgb = xgboost.XGBRegressor(**params_xgb, n_jobs=-1, enable_categorical=True)
            # final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model_xgb)

            # Add validation checks
            # if X_train.isnull().any().any() or y_train.isnull().any():
            #     raise ValueError("Training data contains NaN values")

            # print(f'Trial {proj_utils_plots.beautify(str(trial.number))} Scoring Starts...')
            cv_split_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # Perform cross-validation
            mse_scores = []
            for train_idx, val_idx in cv_split_kf.split(X_train_features, y_train):
                X_tr, X_vl = X_train_features[train_idx], X_train_features[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                # Fit the model and make predictions
                model_xgb.fit(X_tr, y_tr)
                y_vl_pred = model_xgb.predict(X_vl)

                # Validate predictions
                if np.isnan(y_vl_pred).any():
                    raise ValueError("Model generated NaN predictions")

                # Calculate MSE
                mse = mean_squared_error(y_vl, y_vl_pred)
                mse_scores.append(mse)

            score = np.mean(mse_scores)
            # Log to MLflow
            mlflow.log_params(params_xgb)
            mlflow.log_metric("mse", score)
            mlflow.log_metric("rmse", math.sqrt(score))

            # Print the results
            # print("MSE scores:", mse_scores)
            # print("Mean MSE score:", score)

        return score

    optuna.logging.set_verbosity(LOG_OPTUNA_RUN_LEVEL)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # creation of Optuna study
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
        # optimise the study
        study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mse", study.best_value)
        mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

        # print(
        #     f"Best trial: {proj_utils_plots.beautify(study.best_trial)} with value: {study.best_value}"
        # )

        # Log tags
        mlflow.set_tags(
            tags={
                "project": PROJECT_NAME,
                "optimizer_engine": "optuna"
            }
        )

        model = xgboost.XGBRegressor(**study.best_params)
        # final_pipe = proj_utils_feat_engg.create_final_pipeline(pproc_pipe, model)
        model.fit(X_train_features, y_train)
        y_val_pred = model.predict(X_val_features)

        # Log the residual plot
        residuals = proj_utils_plots.plot_residuals(y_val_pred, y_val)
        mlflow.log_figure(residuals, "residuals.png")

        # Log the feature importance's plot
        # importances = proj_utils_plots.plot_feature_importance(model, booster=study.best_params.get("booster"))
        # mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_r2 = r2_score(y_val, y_val_pred)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        artefact_path = artefact_path

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artefact_path,
            input_example=X_train_features[:3]
        )

    return study

def run_hyperparam_tuning_xgb_exp(X_train_features, y_train, X_val_features, y_val, experiment, run_name, num_trials):
    def optuna_objective(trial):
        # Create a new experiment for each trial
        with experiment.train():
            params_xgb = {
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 2),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9)
            }

            model_xgb = xgboost.XGBRegressor(**params_xgb, n_jobs=-1, enable_categorical=True)

            cv_split_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            # Perform cross-validation
            mse_scores = []
            for train_idx, val_idx in cv_split_kf.split(X_train_features, y_train):
                X_tr, X_vl = X_train_features[train_idx], X_train_features[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]

                model_xgb.fit(X_tr, y_tr)
                y_vl_pred = model_xgb.predict(X_vl)

                if np.isnan(y_vl_pred).any():
                    raise ValueError("Model generated NaN predictions")

                mse = mean_squared_error(y_vl, y_vl_pred)
                mse_scores.append(mse)

            score = np.mean(mse_scores)

            # Log parameters and metrics to Comet
            experiment.log_parameters(params_xgb)
            experiment.log_metric("cv_mse", score)
            experiment.log_metric("cv_rmse", math.sqrt(score))

        return score

    # Set experiment name
    experiment.set_name(run_name)

    # Log tags
    list_of_tags = [f"project:{PROJECT_NAME}", "optimizer_engine:optuna"]
    experiment.add_tags(list_of_tags)

    # Create and optimize Optuna study
    optuna.logging.set_verbosity(LOG_OPTUNA_RUN_LEVEL)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE))
    study.optimize(optuna_objective, n_trials=num_trials, callbacks=[champion_callback])

    # Log best parameters and metrics
    experiment.log_parameters(study.best_params)
    experiment.log_metric("best_cv_mse", study.best_value)
    experiment.log_metric("best_cv_rmse", math.sqrt(study.best_value))

    # # Train final model with best parameters
    # model = xgboost.XGBRegressor(**study.best_params)
    # model.fit(X_train_features, y_train)
    # y_val_pred = model.predict(X_val_features)
    #
    # # Create and log residual plot
    # residuals = proj_utils_plots.plot_residuals(y_val_pred, y_val)
    # experiment.log_figure("residuals.png", residuals)
    #
    # # Log validation metrics
    # val_mse = mean_squared_error(y_val, y_val_pred)
    # val_rmse = math.sqrt(val_mse)
    # val_r2 = r2_score(y_val, y_val_pred)
    # experiment.log_metric("val_mse", val_mse)
    # experiment.log_metric("val_rmse", val_rmse)
    # experiment.log_metric("val_r2", val_r2)

    # Log model
    # model_file_name = f"{run_name}.pkl"
    # model_path = f"{PATH_OUT_MODELS}{model_file_name}"
    # save_model(model_file_name, PATH_OUT_MODELS, model)
    # experiment.log_model(run_name, model_path)

    return study

def load_optimised_model(model_path: str):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = joblib.load(model_path)
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("This might be due to version mismatch or corrupted file")
        raise

def train_optimal_model(optimised_study, train_features, train_label, experiment):
    model = xgboost.XGBRegressor(**optimised_study.best_params)
    model.fit(train_features, train_label)

    train_preds = model.predict(train_features)
    residuals = proj_utils_plots.plot_residuals(train_preds, train_label)
    experiment.log_figure("residuals.png", residuals)

    # Log metrics
    train_mse = mean_squared_error(train_label, train_preds)
    train_rmse = math.sqrt(train_mse)
    train_r2 = r2_score(train_label, train_preds)
    experiment.log_metric("train_mse", train_mse)
    experiment.log_metric("train_rmse", train_rmse)
    experiment.log_metric("train_r2", train_r2)

    return model