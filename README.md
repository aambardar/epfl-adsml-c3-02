# House Prices — EPFL ADSML C3

Regression project for the EPFL Applied Data Science and Machine Learning course (Challenge 3).
The goal is to predict residential property sale prices using the housing dataset.

---

## Dataset

**Source:** House prices dataset assembled and published by Dean De Cock.

- 2,930 observations across 82 variables
- Target variable: `SalePrice`

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original train and test CSV files
│   └── processed/            # Intermediate processed data (if any)
├── notebooks/
│   └── c3_submission.ipynb   # Main notebook having end-to-end pipeline
├── outputs/
│   ├── figures/              # Saved plots
│   ├── logs/                 # Application log and MLflow tracking database
│   ├── models/               # Saved model artefacts
│   └── submissions/          # Prediction files
└── src/
    ├── config/
    │   └── settings.py       # All project-wide constants, configs and paths
    ├── data/
    │   └── loader.py         # Data loading and train/test merging
    ├── features/
    │   └── engineering.py    # Feature engineering and sklearn pipeline
    ├── models/
    │   └── trainer.py        # Model training, CV evaluation, MLflow logging
    ├── visualisation/
    │   └── plots.py          # All plotting functions
    └── utils/
        ├── io.py             # File saving helpers
        └── logging.py        # Logging setup (with optional rich support)
```

---

## Approach

1. **Data loading** — train and test sets are merged for consistent preprocessing, then split back before training
2. **Feature engineering** — custom sklearn pipeline with:
   - Ordinal encoding for quality/condition features (defined in `settings.py`)
   - One-hot encoding for low-cardinality nominal features
   - Target encoding for high-cardinality nominal features
   - Year-to-age transformation via a custom `YearTransformer`
   - Median imputation for numerical nulls, constant fill for categorical nulls
3. **Models**
   - Baseline: mean predictor (`DummyRegressor`)
   - Simple model: `LinearRegression` on top-k MI-scored features
   - Lasso regression with Optuna hyperparameter tuning
   - XGBoost with Optuna hyperparameter tuning
4. **Evaluation** — 5-fold cross-validation, RMSE (log scale) and MAE (dollars); held-out validation set scoring
5. **Experiment tracking** — all runs, parameters, metrics, and artefacts logged to MLflow

---

## How to Run

Open and run `notebooks/c3_submission.ipynb` top to bottom.

Before starting a new round of experiments, increment the run version in `src/config/settings.py`:

```python
MODEL_RUN_VERSION = 4   # increment for each new experiment batch
```

This version number is embedded in the MLflow run name and artefact paths, making it easy to
distinguish runs across different experiment iterations.

---

## Experiment Tracking

Runs are tracked in a local SQLite database at `outputs/logs/mlflow.db`.

To view all runs, metrics, and artefacts in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///outputs/logs/mlflow.db --port 5000
```

Then open `http://127.0.0.1:5000` in a browser. The Compare view lets you plot any metric
against any parameter across runs.

---

## Key Configuration

All project-wide settings live in `src/config/settings.py`:

| Constant | Purpose |
|---|---|
| `MODEL_RUN_VERSION` | Increment before each new experiment batch |
| `OPTUNA_TRIAL_COUNT` | Number of Optuna trials per tuning run |
| `VALIDATION_SIZE` | Fraction of training data held out for validation |
| `RANDOM_STATE` | Global random seed |
| `ORDINAL_CATEGORIES` | Ordered category lists for ordinal encoding |
| `MLFLOW_TRACKING_URI` | Path to the MLflow SQLite database |
