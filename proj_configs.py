import os

# TODO: update project name
# Project name
# PROJECT_NAME = 'EPFL-ADSML-C3'
PROJECT_NAME = 'KAGGLE-HOUSE-PRICES'

# OS path configuration
PATH_DATA = 'data/'
PATH_CONTENT = 'content/'
PATH_SRC = 'src/'
PATH_OUTPUT = 'out/'
PATH_OUT_LOGS = 'out/logs/'
PATH_OUT_MODELS = 'out/models/'
PATH_OUT_SUBMISSIONS = 'out/submissions/'
PATH_OUT_FEATURES = 'out/features/'
PATH_OUT_PREDICTIONS = 'out/predictions/'
PATH_OUT_VISUALS = 'plots/'

# Logging configurations
LOG_FILE = os.path.join(PATH_OUT_LOGS, f"{PROJECT_NAME}_application.log")
LOG_ROOT_LEVEL = 'DEBUG'
LOG_FILE_LEVEL = 'DEBUG'
LOG_CONSOLE_LEVEL = 'INFO'

# TODO: update data file configs
# Data files
TRAIN_FILE = os.path.join(PATH_DATA, 'kaggle-house-prices-train.csv')
TEST_FILE = os.path.join(PATH_DATA, 'kaggle-house-prices-test.csv')

# Stylesheet configurations
MPL_STYLE_FILE = os.path.join(PATH_CONTENT, 'custom_mpl_stylesheet.mplstyle')

# Feature Engineering configuration
NUMERICAL_IMPUTATION_STRATEGY = 'mean'  # Options: 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # Options: 'most_frequent', 'constant'

# Path for saving models
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILENAME = 'trained_model.pkl'
BEST_MODEL_PATH = os.path.join(MODEL_PATH, MODEL_FILENAME)

# Other configurations
RANDOM_STATE = 43
VALIDATION_SIZE = 0.2  # For train-test split
OPTUNA_TRIAL_COUNT = 200
MODEL_RUN_VERSION = 0.08
CATEGORICAL_CARDINALITY_THRESHOLD_ABS = 20
CATEGORICAL_CARDINALITY_THRESHOLD_PCT = 0.1