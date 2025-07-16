# data_loader.py
import pandas as pd

from proj_utils_logging import get_logger
logger = get_logger()

def load_data(data_file_path, is_col_refactor_needed):
    try:
        logger.debug("START ...")
        logger.info(f'Loading data from path: {data_file_path}')
        df_data = pd.read_csv(data_file_path)
        logger.info(f'Loaded train data with shape: {df_data.shape}')
        logger.info(f'Successfully loaded data from path: {data_file_path}')
        if is_col_refactor_needed:
            df_data = refactor_col_names(df_data)
        logger.debug("... FINISH")
        return df_data
    except Exception as e:
        logger.error(f'Failed to load data from {data_file_path}: {str(e)}')
        raise

def refactor_col_names(df: pd.DataFrame):
    logger.debug("START ...")
    df.columns = df.columns.str.replace(' ', '')
    df = df.rename(columns={'YearRemod/Add': 'YearRemodAdd'})
    logger.debug("... FINISH")
    return df

def merge_train_test_data(train_df: pd.DataFrame, test_df: pd.DataFrame, cols_to_drop: list, target_label: str):
    try:
        logger.debug("START ...")
        logger.info(f'Merging train data shape: {train_df.shape} and test data shape: {test_df.shape}')

        # Create indicator column
        train_df['is_train'] = 1
        test_df['is_train'] = 0

        # Drop specified columns
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
            test_df = test_df.drop(columns=cols_to_drop, errors='ignore')
            logger.info(f'Dropped columns: {cols_to_drop}')

        # Store target variable separately for train data
        train_target = train_df[target_label].copy()
        train_df = train_df.drop(columns=[target_label])

        # Merge dataframes
        merged_df = pd.concat([train_df, test_df], axis=0)
        logger.info(f'Merged data shape: {merged_df.shape}')
        logger.debug("... FINISH")

        return merged_df, train_target
    except Exception as e:
        logger.error(f'Failed to merge train and test data: {str(e)}')
        raise
