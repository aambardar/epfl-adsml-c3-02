import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from src.config.settings import CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS, CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT

from src.utils.logging import get_logger
logger = get_logger()

# Custom Transformer
class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Store original column names during fit
        if hasattr(X, 'columns'):
            self.original_columns = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            # If X is a numpy array, try to get column names from the ColumnTransformer
            # This assumes the transformer is part of a ColumnTransformer
            if hasattr(self, 'feature_names_in_'):
                self.original_columns = self.feature_names_in_
            else:
                self.original_columns = [f'col_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X, y=None):
        # Convert numpy array to pandas DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        else:
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        current_year = pd.Timestamp.now().year
        columns_to_drop = []

        for base_col_name in X_transformed.columns:
            transformed_col_name = base_col_name + '_years_old'
            X_transformed[transformed_col_name] = current_year - X_transformed[base_col_name]
            columns_to_drop.append(base_col_name)
            # print(f"Processed column: {base_col_name} -> {transformed_col_name}")

        # Drop all original columns at once (more efficient than dropping inside loop)
        X_transformed.drop(columns_to_drop, axis=1, inplace=True)

        # store feature names
        self.feature_names = X_transformed.columns
        return X_transformed.values  # Return numpy array to maintain scikit-learn compatibility

    def set_output(self, *, transform=None):
        super().set_output(transform=transform)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.feature_names


def get_cols_as_tuple(feat_categories):
    logger.debug("START ...")
    cols_num_continuous = feat_categories.get("numerical_continuous")
    n_num_continuous = len(cols_num_continuous)

    cols_num_discrete = feat_categories.get("numerical_discrete")
    n_num_discrete = len(cols_num_discrete)

    cols_cat_nominal = feat_categories.get("categorical_nominal")
    n_cat_nominal = len(cols_cat_nominal)

    cols_cat_ordinal = feat_categories.get("categorical_ordinal")
    n_cat_ordinal = len(cols_cat_ordinal)

    cols_object = feat_categories.get("object")
    n_object = len(cols_object)

    cols_temporal = feat_categories.get("temporal")
    n_temporal = len(cols_temporal)

    cols_binary = feat_categories.get("binary")
    n_binary = len(cols_binary)

    cols_low_cardinality = feat_categories.get("low_cardinality")
    n_low_cardinality = len(cols_low_cardinality)

    logger.debug("... FINISH")

    return (cols_num_continuous, n_num_continuous, cols_num_discrete, n_num_discrete, cols_cat_nominal, n_cat_nominal,
            cols_cat_ordinal, n_cat_ordinal, cols_object, n_object, cols_temporal, n_temporal, cols_binary, n_binary, cols_low_cardinality, n_low_cardinality)


def _is_low_cardinality(col, df, threshold_type, n_cat_threshold):
    """Returns True if column cardinality is below the threshold."""
    if threshold_type == CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS:
        return df[col].nunique() <= n_cat_threshold
    count = df[col].count()
    if count == 0:
        return False
    return df[col].nunique() / count <= n_cat_threshold

def classify_columns(df, n_cat_threshold, threshold_type='ABS', cols_to_ignore=None, temporal_cols_name_pattern=None,
                     ordinal_cols=None):
    """
      Classify each column of a DataFrame into one of eight feature types based
      on dtype, cardinality, and caller-supplied metadata.

      Classification priority (applied in order):
          1. Temporal — datetime/timedelta dtype, or name matches a temporal pattern
          2. Ordinal — column name explicitly listed in ordinal_cols
          3. Binary — boolean dtype, or exactly 2 unique values
          4. Numerical — numeric dtype; further split into discrete (integer + low
                         cardinality) vs continuous
          5. Categorical nominal — pandas Categorical dtype
          6. String/object — object dtype; split into nominal (low cardinality) vs object
          7. Fallback — anything unmatched is cast to object with a warning

      Another derived group, low_cardinality, is a subset of categorical_nominal
      columns that fall below the cardinality threshold.

      Parameters:
          df (pd.DataFrame): Input DataFrame whose columns are to be classified.
          n_cat_threshold (int | float): Cardinality threshold. Interpreted as an
              absolute unique-value count when threshold_type='ABS', or as a
              fraction of non-null rows when threshold_type='PCT'.
          threshold_type (str): How to apply n_cat_threshold. 'ABS' for absolute
              count, 'PCT' for percentage. Defaults to 'ABS'.
          cols_to_ignore (list[str] | None): Columns to skip entirely (e.g. the
              target variable). Defaults to None (no columns skipped).
          temporal_cols_name_pattern (list[str] | None): Substrings to match
              against column names to identify temporal columns by name (e.g.
              ['Year', 'Mo']). Defaults to None.
          ordinal_cols (list[str] | None): Column names to explicitly classify as
              categorical_ordinal regardless of dtype. Defaults to None.

      Returns:
          dict[str, list[str]]: Keys are feature type labels, values are lists of
              column names assigned to that type:
              'numerical_continuous', 'numerical_discrete', 'categorical_nominal',
              'categorical_ordinal', 'object', 'temporal', 'binary', 'low_cardinality'

      Raises:
          ValueError: If threshold_type is not 'ABS' or 'PCT'.
    """
    logger.debug("START ...")

    cols_to_ignore = cols_to_ignore or []
    temporal_cols_name_pattern = temporal_cols_name_pattern or []
    ordinal_cols = ordinal_cols or []

    valid_threshold_types = {CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS, CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT}
    if threshold_type not in valid_threshold_types:
        raise ValueError(f"threshold_type must be one of {valid_threshold_types}, got '{threshold_type}'")

    logger.debug(f"Input: {len(df.columns)} columns | threshold_type={threshold_type} | n_cat_threshold = {n_cat_threshold}")
    logger.debug(f"Ignoring {len(cols_to_ignore)} cols | {len(ordinal_cols)} ordinal cols pre-assigned | temporal patterns = {temporal_cols_name_pattern}")

    feature_types = {
        'numerical_continuous': [],
        'numerical_discrete': [],
        'categorical_nominal': [],
        'categorical_ordinal': [],
        'object': [],
        'temporal': [],
        'binary': [],
        'low_cardinality': []
    }

    for column in df.columns:
        # Skip target variable
        if column in cols_to_ignore:
            logger.debug(f"  '{column}' → skipped (in cols_to_ignore)")
            continue
        assigned = None
        # Check if a column is temporal based on name or type
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[column])
        is_timedelta = pd.api.types.is_timedelta64_dtype(df[column])
        has_temporal_name = any(pattern in column for pattern in temporal_cols_name_pattern)

        # Check if a column is ordinal based on name
        has_ordinal_match = column in ordinal_cols

        # Check if a column is boolean based on type
        is_boolean = pd.api.types.is_bool_dtype(df[column])

        # Check if a column is numerical or integer based on type
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        is_integer = pd.api.types.is_integer_dtype(df[column])

        # Check if a column is categorical based on type
        is_categorical = isinstance(df[column].dtype, pd.CategoricalDtype)

        # Check if a column is string or object based on type
        is_string_object = pd.api.types.is_string_dtype(df[column])  # Checks both object and string dtypes

        # Temporal features
        if is_datetime or is_timedelta or has_temporal_name:
            assigned = 'temporal'

        # Ordinal features
        elif has_ordinal_match:
            assigned = 'categorical_ordinal'

        # Binary features
        elif df[column].nunique() == 2 or is_boolean:
            assigned = 'binary'

        # Numerical features
        elif is_numeric:
            # Basic heuristic: if the number of unique values is small relative to the not null values
            if is_integer:
                if _is_low_cardinality(column, df, threshold_type, n_cat_threshold):
                    assigned = 'numerical_discrete'
                else:
                    assigned = 'numerical_continuous'
            else:
                assigned = 'numerical_continuous'

        # Categorical features
        elif is_categorical:
            assigned = 'categorical_nominal'

        elif is_string_object:
            # Basic heuristic: if the number of unique values is small relative to the not null values
            if _is_low_cardinality(column, df, threshold_type, n_cat_threshold):
                assigned = 'categorical_nominal'
            else:
                assigned = 'object'
        else:
            logger.warning(f'Matching Col Type Not Found for: {column}, so casting as Object type.')
            assigned = 'object'

        feature_types[assigned].append(column)
        logger.debug(f"  '{column}' → {assigned}")

    feature_types['low_cardinality'] = [
        cname for cname in df.columns
        if cname not in cols_to_ignore
           and df[cname].dtype == "object"
           and _is_low_cardinality(cname, df, threshold_type, n_cat_threshold)
    ]

    logger.info("Feature classification summary:")
    for ftype, features in feature_types.items():
        logger.info(f"  {ftype:<25} : {len(features):>3} cols")

    # Full feature names only in file (DEBUG)
    for ftype, features in feature_types.items():
        logger.debug(f"  {ftype}: {features}")

    logger.debug("... FINISH")
    return feature_types


def get_cardinality_df(df):
    # Create a cardinality DF that captures pct of not null and null values, along with pct of unique values
    logger.debug("START ...")
    null_counts = df.isnull().sum()
    df_cardinality = pd.DataFrame({
        'col_name': df.columns,
        'notnull_pct': (1 - null_counts / df.shape[0]).round(3).values,
        'null_pct': (null_counts / df.shape[0]).round(3).values,
        'unique_pct': (df.nunique() / df.shape[0]).round(3).values,
    })
    df_cardinality = df_cardinality.sort_values('null_pct').reset_index(drop=True)
    logger.debug("... FINISH")
    return df_cardinality


def create_pproc_pipeline(cols_num, cols_cat, cols_temporal):
    logger.debug("START ...")
    # imputer to replace nulls with the most frequent value
    imputer_cat_freq = SimpleImputer(strategy='most_frequent')
    # imputer to replace nulls with the most frequent value
    imputer_num_med = SimpleImputer(strategy='median')
    # imputer to replace nulls with the mean value
    imputer_num_mean = SimpleImputer(strategy='mean')
    # one-hot encoding
    encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # feature scaling
    scaler_std = StandardScaler()
    # custom transformer to replace Year with age
    trans_year = YearTransformer()

    # custom transformer to split one date time column into its constituting elements (data, month and year)
    # transform_date = DateTransformer()

    numeric_columns = cols_num
    categorical_columns = cols_cat
    temporal_columns = cols_temporal

    # Numeric pipeline with missing value imputation and scaling
    pipe_numeric_transformer = Pipeline(steps=[
        # ('imputer', imputer_num_med),
        ('imputer', imputer_num_mean),
        ('scaler', scaler_std)
    ])

    # Categorical pipeline with missing value imputation and one-hot encoding
    pipe_categorical_transformer = Pipeline(steps=[
        ('imputer', imputer_cat_freq),
        ('encoder', encoder_ohe)
    ])

    # Temporal pipeline to replace year with age
    pipe_temporal_transformer = Pipeline(steps=[
        # ('imputer', imputer_num_med),
        ('imputer', imputer_cat_freq),
        ('trans_tempo', trans_year)
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', pipe_numeric_transformer, numeric_columns),
            ('cat', pipe_categorical_transformer, categorical_columns),
            ('tempo', pipe_temporal_transformer, temporal_columns)
        ],
        remainder='passthrough'
    )
    logger.debug("... FINISH")
    return preprocessor


def create_final_pipeline(pproc_pipe, model):
    logger.debug("START ...")
    # Create a full pipeline with preprocessing and model
    final_pipe = Pipeline(steps=[
        ('preprocessor', pproc_pipe),
        ('regressor', model)  # alpha parameter controls regularization strength
    ])
    logger.debug("... FINISH")
    return final_pipe

def get_final_features(final_pipeline, X_train):
    logger.debug("START ...")
    feature_names = []
    column_names = []
    pproc = final_pipeline.named_steps['preprocessor']

    for i, (name, trans, column) in enumerate(pproc.transformers_):
        print(f'Transformer#{i + 1} name is:{proj_utils_plots.beautify(str(name))}')
        if type(trans) is Pipeline:
            print('\t Is a Pipeline')
            trans = trans.steps[-1][1]
        else:
            print('\t Isn\'t a Pipeline')
        if hasattr(trans, 'get_feature_names_out'):
            print('\t Has get_feature_names_out')
            tmp_feature_names = trans.get_feature_names_out(column)
            feature_names.extend(tmp_feature_names)
            column_names.extend(column)
            print(
                f'\t Transformer input = {proj_utils_plots.beautify(str(len(column)), 1)} and output = {proj_utils_plots.beautify(str(len(tmp_feature_names)), 1)}')
        elif hasattr(trans, 'get_feature_names'):
            print('\t Has get_feature_names')
            tmp_feature_names = trans.get_feature_names(column)
            feature_names.extend(tmp_feature_names)
            column_names.extend(column)
            print(
                f'\t Transformer input = {proj_utils_plots.beautify(str(len(column)), 1)} and output = {proj_utils_plots.beautify(str(len(tmp_feature_names)), 1)}')
        else:
            print('\t Doesn\'t have get_feature_names or get_feature_names_out')
            if name == 'remainder' and trans == 'passthrough':
                print('\t > It\'s remainder passthrough')
                tmp_remainder_names = set(X_train.columns) - set(column_names)
                feature_names.extend(tmp_remainder_names)
                column_names.extend(column)
                print(
                    f'\t Transformer input = {proj_utils_plots.beautify(str(len(column)), 1)} and output = {proj_utils_plots.beautify(str(len(column)), 1)}')
            else:
                print('\t > Not a remainder passthrough')
                tmp_feature_names = column
                feature_names.extend(tmp_feature_names)
                column_names.extend(column)
                print(
                    f'\t Transformer input = {proj_utils_plots.beautify(str(len(column)), 1)} and output = {proj_utils_plots.beautify(str(len(tmp_feature_names)), 1)}')

    print(
        f'\nThe total feature space has: {proj_utils_plots.beautify(str(len(feature_names)))} features. Their names being:\n{proj_utils_plots.beautify(str(feature_names), 2)}')
    return column_names, feature_names
    logger.debug("... FINISH")
