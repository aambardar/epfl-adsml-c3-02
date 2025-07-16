"""
Main script for House Prices Prediction
This script replaces the Jupyter notebook functionality in a modular way
"""

import os
import numpy as np
import pandas as pd
import warnings
import logging

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import comet_ml

# Local imports
import proj_configs as configs
import proj_utils as utils
import proj_utils_data_loader as data_loader
import proj_utils_feat_engg as feat_engg
import proj_utils_model as model_utils
import proj_utils_plots as plot_utils
import proj_utils_logging as log_utils


class HousePricePredictor:
    def __init__(self):
        self.setup_environment()
        self.logger = log_utils.get_logger()
        self.logger.debug(f"Initialised logging for {configs.PROJECT_NAME} project.")

    def setup_environment(self):
        """Initialize environment settings and configurations"""
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("mlflow").setLevel(logging.ERROR)

    def load_data(self):
        """Load and prepare the training and test datasets"""
        self.logger.debug("START ...")
        self.df_raw_train = data_loader.load_data(configs.TRAIN_FILE, True)
        self.df_raw_test = data_loader.load_data(configs.TEST_FILE, True)

        # Define column categories
        self.insignificant_cols = ['Id']
        self.target_col = 'SalePrice'
        self.ignorables_cols = self.insignificant_cols + [self.target_col]
        self.ordinal_cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual',
                             'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
                             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                             'GarageCond', 'PoolQC']
        self.temporal_cols_name_pattern = ['Yr', 'Year']
        self.logger.debug("... FINISH")

    def prepare_data(self):
        """Prepare and process the data for modeling"""
        self.logger.debug("START ...")
        # Merge train and test data
        self.df_raw_all, self.df_raw_target = data_loader.merge_train_test_data(
            self.df_raw_train,
            self.df_raw_test,
            self.insignificant_cols,
            self.target_col
        )

        # Split into train and test
        self.df_train = self.df_raw_all[self.df_raw_all['is_train'] == 1].iloc[:, :-1]
        self.df_test = self.df_raw_all[self.df_raw_all['is_train'] == 0].iloc[:, :-1]

        # Classify features
        self.feature_categories = feat_engg.classify_columns(
            df=self.df_train,
            n_cat_threshold=configs.CATEGORICAL_CARDINALITY_THRESHOLD_ABS,
            threshold_type='ABS',
            cols_to_ignore=self.ignorables_cols,
            temporal_cols_name_pattern=self.temporal_cols_name_pattern,
            ordinal_cols=self.ordinal_cols
        )

        # Get column categories
        (self.cols_num_continuous, _, self.cols_num_discrete, _,
         self.cols_cat_nominal, _, self.cols_cat_ordinal, _,
         self.cols_object, _, self.cols_temporal, _,
         self.cols_binary, _, self.cols_low_cardinality, _) = feat_engg.get_cols_as_tuple(self.feature_categories)
        self.logger.debug("... FINISH")

    def create_train_val_split(self):
        """Create training and validation splits"""
        self.logger.debug("START ...")
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.df_train,
            self.df_raw_target,
            test_size=configs.VALIDATION_SIZE,
            random_state=configs.RANDOM_STATE
        )
        self.logger.debug("... FINISH")

    def setup_preprocessing(self):
        """Setup the preprocessing pipeline"""
        self.logger.debug("START ...")
        self.num_columns = self.cols_num_continuous
        self.cat_columns = (self.cols_cat_nominal + self.cols_cat_ordinal +
                            self.cols_num_discrete + self.cols_binary + self.cols_object)
        self.tempo_columns = self.cols_temporal

        self.pproc_pipe = feat_engg.create_pproc_pipeline(
            self.num_columns,
            self.cat_columns,
            self.tempo_columns
        )
        self.pproc_pipe_full_train = feat_engg.create_pproc_pipeline(
            self.num_columns,
            self.cat_columns,
            self.tempo_columns
        )
        self.logger.debug("... FINISH")

    def transform_data(self):
        """Transform the data using the preprocessing pipeline"""
        self.logger.debug("START ...")
        self.X_train_transformed = self.pproc_pipe.fit_transform(self.X_train)
        self.X_val_transformed = self.pproc_pipe.transform(self.X_val)
        self.y_train_transformed = self.y_train.to_numpy()
        self.y_val_transformed = self.y_val.to_numpy()

        self.train_transformed = self.pproc_pipe_full_train.fit_transform(self.df_train)
        self.train_labels_transformed = self.df_raw_target.to_numpy()
        self.logger.debug("... FINISH")

    def tune_model(self):
        """Train the model using XGBoost with hyperparameter optimization"""
        self.logger.debug("START ...")
        comet_experiment = comet_ml.Experiment()
        run_base_name = 'xgb-house-prices'
        run_count = configs.MODEL_RUN_VERSION
        self.run_name = f'{run_base_name}-{run_count}'
        self.logger.info(f"Run name starts: {self.run_name} under Experiment: {comet_experiment.project_name}/{comet_experiment.workspace}")
        try:
            self.optimized_study = model_utils.run_hyperparam_tuning_xgb_exp(
                self.X_train_transformed,
                self.y_train_transformed,
                self.X_val_transformed,
                self.y_val_transformed,
                comet_experiment,
                self.run_name,
                configs.OPTUNA_TRIAL_COUNT
            )
            self.logger.info(f"Optimised study best params: {self.optimized_study.best_params}")
            self.tuned_model = model_utils.train_optimal_model(self.optimized_study, self.train_transformed, self.train_labels_transformed, comet_experiment)
            model_file_name = f'{self.run_name}.pkl'
            model_utils.save_model(model_file_name, configs.PATH_OUT_MODELS, self.tuned_model)
            comet_experiment.log_model(self.run_name, f"{configs.PATH_OUT_MODELS}{model_file_name}")
        finally:
            comet_experiment.end()
            self.logger.info(f"Run name ends: {self.run_name}")
        self.logger.debug("... FINISH")

    def load_model(self):
        self.logger.debug("START ...")
        self.loaded_model = model_utils.load_optimised_model(f"{configs.PATH_OUT_MODELS}{self.run_name}.pkl")
        self.logger.debug("... FINISH")

    def make_predictions(self):
        """Make predictions on test data"""
        self.logger.debug("START ...")
        self.test_transformed = self.pproc_pipe_full_train.transform(self.df_test)
        self.test_predictions = self.tuned_model.predict(self.test_transformed)
        self.logger.debug("... FINISH")

    def save_predictions(self):
        """Save predictions to a CSV file"""
        self.logger.debug("START ...")
        submission_df = pd.DataFrame({
            'Id': self.df_raw_test.Id,
            'SalePrice': self.test_predictions
        })
        submission_file_name = f'{self.run_name}_submission.csv'
        submission_file = os.path.join(
            configs.PATH_OUT_SUBMISSIONS,
            submission_file_name
        )
        submission_df.to_csv(submission_file, index=False)
        self.logger.info(f"Predictions saved to {submission_file}")
        self.logger.debug("... FINISH")

def main():
        predictor = HousePricePredictor()
        predictor.load_data()
        predictor.prepare_data()
        predictor.create_train_val_split()
        predictor.setup_preprocessing()
        predictor.transform_data()
        predictor.tune_model()
        predictor.load_model()
        predictor.make_predictions()
        predictor.save_predictions()
        # predictor.evaluate_model()

if __name__ == "__main__":
    main()