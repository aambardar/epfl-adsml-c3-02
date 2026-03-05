import os
from pathlib import Path

# TODO: update project name
# Project name
PROJECT_NAME = 'ADSML-C3-V2-HOUSE-PRICES'
# Anchor to the project root (3 levels up from src/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All paths derived from root
PATH_DATA             = PROJECT_ROOT / 'data'
PATH_DATA_RAW         = PROJECT_ROOT / 'data' / 'raw'
PATH_DATA_PROCESSED   = PROJECT_ROOT / 'data' / 'processed'
PATH_OUTPUT           = PROJECT_ROOT / 'outputs'
PATH_OUT_LOGS         = PROJECT_ROOT / 'outputs' / 'logs'
PATH_OUT_MODELS       = PROJECT_ROOT / 'outputs' / 'models'
PATH_OUT_SUBMISSIONS  = PROJECT_ROOT / 'outputs' / 'submissions'
PATH_OUT_FEATURES     = PROJECT_ROOT / 'outputs' / 'features'
PATH_OUT_VISUALS      = PROJECT_ROOT / 'outputs' / 'figures'

# Logging configurations
LOG_FILE     = PATH_OUT_LOGS / f"{PROJECT_NAME}_application.log"
LOG_ROOT_LEVEL = 'DEBUG'
LOG_FILE_LEVEL = 'DEBUG'
LOG_CONSOLE_LEVEL = 'INFO'

# TODO: update data file configs
# Data files
TRAIN_FILENAME = 'house-prices.csv'
TEST_FILENAME = 'house-prices-test.csv'
TRAIN_FILE = PATH_DATA_RAW / TRAIN_FILENAME
TEST_FILE = PATH_DATA_RAW / TEST_FILENAME

# Stylesheet configurations
MPL_STYLE_FILE    = PROJECT_ROOT / 'src' / 'config' / 'custom_mpl_stylesheet.mplstyle'

# Feature Engineering configuration
NUMERICAL_IMPUTATION_STRATEGY = 'mean'  # Options: 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # Options: 'most_frequent', 'constant'

# Path for saving models
MODEL_FILENAME = 'trained_model.pkl'
BEST_MODEL_PATH   = PATH_OUT_MODELS / MODEL_FILENAME

# Other configurations
RANDOM_STATE = 43
VALIDATION_SIZE = 0.2  # For train-test split
OPTUNA_TRIAL_COUNT = 200
MODEL_RUN_VERSION = 0.08
CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS = 'ABS'
CATEGORICAL_CARDINALITY_THRESHOLD_ABS_VAL = 20
CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT = 'PCT'
CATEGORICAL_CARDINALITY_THRESHOLD_PCT_VAL = 0.1
