"""
engineering.py — Data preprocessing & EDA helper module.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from src.config.settings import CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_ABS, CATEGORICAL_CARDINALITY_THRESHOLD_TYPE_PCT, ORDINAL_CATEGORIES

from src.utils.logging import get_logger
from src.visualisation.plots import beautify

logger = get_logger()


# Custom Transformer: converts year columns to age (reference_year_ - year_value).
class YearTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """Capture column names, input shape, and reference year from training data."""
        # Store column names for consistent numpy input handling in transform
        if hasattr(X, 'columns'):
            self.original_columns_ = X.columns.tolist()
        elif isinstance(X, np.ndarray):
            # Feature names may be injected by ColumnTransformer via feature_names_in_
            if hasattr(self, 'feature_names_in_'):
                self.original_columns_ = list(self.feature_names_in_)
            else:
                self.original_columns_ = [f'col_{i}' for i in range(X.shape[1])]

        # Capture reference year at fit time so all subsequent transforms
        # (train, val, inference) compute age consistently against the same year
        self.reference_year_ = pd.Timestamp.now().year

        # Required by sklearn's check_is_fitted and validation utilities
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """Replace each year column with its age relative to reference_year_."""
        # Use stored column names for numpy input to stay consistent with fit
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.original_columns_)
        else:
            X = pd.DataFrame(X)

        # Compute age for each column and drop the original year column
        X_transformed = X.copy()
        for col in self.original_columns_:
            X_transformed[col + '_years_old'] = self.reference_year_ - X_transformed[col]
        X_transformed = X_transformed.drop(columns=self.original_columns_)

        return X_transformed.values  # Return numpy array to maintain scikit-learn compatibility

    def get_feature_names_out(self, input_features=None):
        """Return output feature names; derived from original_columns_ set during fit."""
        return np.array([f'{col}_years_old' for col in self.original_columns_])

def print_feature_expansion(pproc_pipe):
    """
    Prints a formatted mapping of input columns to output features for each
    transformer branch in the fitted ColumnTransformer.

    Parameters
    ----------
    pproc_pipe : fitted sklearn ColumnTransformer
    """
    logger.debug("START ...")
    total_in, total_out = 0, 0

    for branch_name, transformer, columns in pproc_pipe.transformers_:
      if branch_name == 'remainder':
          continue

      print(f"\n  [{branch_name.upper()} branch]")
      print(f"  {'INPUT COLUMN':<30} {'N_OUT':>6}  OUTPUT FEATURE NAMES")
      print(f"  {'-'*80}")

      last_step = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer

      for i, col in enumerate(columns):

          # OHE: 1 input → N outputs (one per category value)
          if isinstance(last_step, OneHotEncoder):
              out_names = [f"{col}_{cat}" for cat in last_step.categories_[i]]
              n_out = len(out_names)
              preview = ', '.join(out_names[:4]) + (f'  ... +{n_out - 4} more' if n_out > 4 else '')

          # OrdinalEncoder: 1-to-1, value replaced with integer rank
          elif isinstance(last_step, OrdinalEncoder):
              out_names = [col]
              n_out = 1
              preview = f"{col}  → integer rank"

          # YearTransformer: 1-to-1, year replaced with age
          elif isinstance(last_step, YearTransformer):
              out_names = [f"{col}_years_old"]
              n_out = 1
              preview = out_names[0]

          # StandardScaler and others: 1-to-1 passthrough with scaling
          else:
              out_names = [col]
              n_out = 1
              preview = col

          print(f"  {col:<30} {n_out:>6}  {preview}")
          total_in += 1
          total_out += n_out

    print(f"\n  {'='*80}")
    print(f"  TOTAL: {total_in} input columns  →  {total_out} output columns")
    logger.debug("... FINISH")


def get_cols_as_tuple(feat_categories):
    """
    Unpack a feature category dict into a flat tuple of (column list, count) pairs.

    Returns one (list, int) pair per category in a fixed order:
    numerical_continuous, numerical_discrete, categorical_nominal,
    categorical_ordinal, object, temporal, binary, low_cardinality.

    Parameters
    ----------
    feat_categories : dict
        Output of classify_columns.

    Returns
    -------
    tuple
        Flat tuple of alternating column lists and counts, 16 elements total.
    """
    logger.debug("START ...")
    category_order = [
        "numerical_continuous", "numerical_discrete", "categorical_nominal",
        "categorical_ordinal", "object", "temporal", "binary", "low_cardinality",
    ]
    result = []
    for key in category_order:
        cols = feat_categories.get(key)
        result.extend([cols, len(cols)])
    return tuple(result)

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
    Classify each column of a DataFrame into one of eight feature types.

    Classification runs in priority order: temporal, ordinal, binary,
    numerical (discrete vs continuous), categorical nominal, string/object,
    then a fallback to object for anything unmatched.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to classify.
    n_cat_threshold : int or float
        Cardinality threshold. Interpreted as absolute unique-value count
        when threshold_type is 'ABS', or as a fraction of non-null rows
        when threshold_type is 'PCT'.
    threshold_type : str
        How to apply n_cat_threshold. 'ABS' for absolute, 'PCT' for percentage.
        Defaults to 'ABS'.
    cols_to_ignore : list of str, optional
        Columns to skip entirely, e.g. the target variable.
    temporal_cols_name_pattern : list of str, optional
        Substrings matched against column names to flag temporal columns by name.
    ordinal_cols : list of str, optional
        Column names to explicitly classify as categorical_ordinal.

    Returns
    -------
    dict
        Keys are feature type labels, values are lists of column names.
        Types: 'numerical_continuous', 'numerical_discrete', 'categorical_nominal',
        'categorical_ordinal', 'object', 'temporal', 'binary', 'low_cardinality'.

    Raises
    ------
    ValueError
        If threshold_type is not 'ABS' or 'PCT'.
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
    """
    Build a cardinality summary DataFrame for all columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with notnull_pct, null_pct, and unique_pct fields,
        sorted ascending by null_pct.
    """
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


def create_pproc_pipeline(cols_num, cols_cat, cols_ord_cat, cols_temporal):
    """
    Build a sklearn ColumnTransformer preprocessing pipeline for four column groups.

    Each group gets its own sub-pipeline:
      - Numeric    : median imputation → standard scaling
      - Categorical: most-frequent imputation → one-hot encoding
      - Ordinal    : most-frequent imputation → ordinal encoding (order from ORDINAL_CATEGORIES)
      - Temporal   : median imputation → year-to-age transformation (YearTransformer)

    Parameters
    ----------
    cols_num : list[str]
        Continuous and discrete numeric columns.
    cols_cat : list[str]
        Nominal categorical columns (low-cardinality strings / pandas Categorical).
    cols_ord_cat : list[str]
        Ordinal categorical columns. Every column must have an entry in
        ORDINAL_CATEGORIES defining the category order (low → high).
    cols_temporal : list[str]
        Year columns to be converted to age by YearTransformer.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted-ready preprocessor; pass to a downstream Pipeline with a model.

    Raises
    ------
    ValueError
        If any column in cols_ord_cat is missing from ORDINAL_CATEGORIES.
    """
    logger.debug("START ...")

    # Validate all ordinal columns have an explicitly defined category order
    missing = [c for c in cols_ord_cat if c not in ORDINAL_CATEGORIES]
    if missing:
        raise ValueError(f"No category order defined for ordinal cols: {missing}")

    # Build the ordered categories list aligned to cols_ord_cat column order
    ordinal_categories = [ORDINAL_CATEGORIES[col] for col in cols_ord_cat]

    # Encoders
    encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_ordinal = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )

    # Scaler
    scaler_std = StandardScaler()

    # Custom transformer: converts year columns to age (current_year - year)
    trans_year = YearTransformer()

    # Numeric pipeline: impute missing with median, then standardise
    pipe_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler_std)
    ])

    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    pipe_categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder_ohe)
    ])

    # Ordinal pipeline: impute missing with most frequent, then ordinal encode
    # preserving the low→high order defined in ORDINAL_CATEGORIES
    pipe_ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder_ordinal)
    ])

    # Temporal pipeline: impute missing with median, then convert year to age
    pipe_temporal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('trans_tempo', trans_year)
    ])

    # Combine all sub-pipelines; drop any column not explicitly assigned to a group
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', pipe_numeric_transformer, cols_num),
            ('cat', pipe_categorical_transformer, cols_cat),
            ('ord', pipe_ordinal_transformer, cols_ord_cat),
            ('tempo', pipe_temporal_transformer, cols_temporal)
        ],
        remainder='passthrough'
    )
    logger.debug("... FINISH")
    return preprocessor


def create_final_pipeline(pproc_pipe, model):
    """
    Wrap a preprocessor and a model into a single sklearn Pipeline.

    Parameters
    ----------
    pproc_pipe : sklearn.compose.ColumnTransformer
        Fitted or unfitted preprocessor.
    model : sklearn estimator
        The regression model to place after preprocessing.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    logger.debug("START ...")
    # Create a full pipeline with preprocessing and model
    final_pipe = Pipeline(steps=[
        ('preprocessor', pproc_pipe),
        ('regressor', model)  # alpha parameter controls regularization strength
    ])
    logger.debug("... FINISH")
    return final_pipe

def get_final_features(final_pipeline, X_train):
    """
    Extract and print input/output feature names for each transformer in the pipeline.

    Iterates over the preprocessor step and calls get_feature_names_out where
    available, printing a summary of input vs output counts per transformer.

    Parameters
    ----------
    final_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing a 'preprocessor' step.
    X_train : pd.DataFrame
        Training data used to fit the pipeline.

    Returns
    -------
    tuple of (list, list)
        feature_names and column_names extracted from the pipeline.
    """
    logger.debug("START ...")
    feature_names = []
    column_names = []
    pproc = final_pipeline.named_steps['preprocessor']

    for i, (name, trans, column) in enumerate(pproc.transformers_):
        print(f'Transformer#{i + 1} name is:{beautify(str(name))}')
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
                f'\t Transformer input = {beautify(str(len(column)), 1)} and output = {beautify(str(len(tmp_feature_names)), 1)}')
        elif hasattr(trans, 'get_feature_names'):
            print('\t Has get_feature_names')
            tmp_feature_names = trans.get_feature_names(column)
            feature_names.extend(tmp_feature_names)
            column_names.extend(column)
            print(
                f'\t Transformer input = {beautify(str(len(column)), 1)} and output = {beautify(str(len(tmp_feature_names)), 1)}')
        else:
            print('\t Doesn\'t have get_feature_names or get_feature_names_out')
            if name == 'remainder' and trans == 'passthrough':
                print('\t > It\'s remainder passthrough')
                tmp_remainder_names = set(X_train.columns) - set(column_names)
                feature_names.extend(tmp_remainder_names)
                column_names.extend(column)
                print(
                    f'\t Transformer input = {beautify(str(len(column)), 1)} and output = {beautify(str(len(column)), 1)}')
            else:
                print('\t > Not a remainder passthrough')
                tmp_feature_names = column
                feature_names.extend(tmp_feature_names)
                column_names.extend(column)
                print(
                    f'\t Transformer input = {beautify(str(len(column)), 1)} and output = {beautify(str(len(tmp_feature_names)), 1)}')

    print(
        f'\nThe total feature space has: {beautify(str(len(feature_names)))} features. Their names being:\n{beautify(str(feature_names), 2)}')
    return column_names, feature_names
    logger.debug("... FINISH")
