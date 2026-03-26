# House Prices — EPFL ADSML C3

Regression project for the EPFL Applied Data Science and Machine Learning course (Challenge 3).
The goal is to predict residential property sale prices using the Ames, Iowa housing dataset.

---

## Dataset

**Source:** Ames, Iowa Assessor's Office — introduced in De Cock (2011) as a modern alternative to the
Boston Housing dataset.

- 2,930 observations across 82 variables (23 nominal, 23 ordinal, 14 discrete, 20 continuous)
- Covers residential property sales in Ames, IA from 2006 to 2010
- Target variable: `SalePrice`

Full variable documentation: https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt

---

## Project Structure

```
.
├── data/
│   ├── raw/                   # Original train and test CSV files
│   └── processed/             # Intermediate processed data (if any)
├── notebooks/
│   └── c3_p2_submission.ipynb # Main notebook — end-to-end pipeline
├── outputs/
│   ├── figures/               # Saved plots
│   ├── logs/                  # Application log and MLflow tracking database
│   ├── models/                # Saved model artefacts
│   └── submissions/           # Kaggle-format prediction files
└── src/
    ├── config/
    │   └── settings.py        # All project-wide constants and paths
    ├── data/
    │   └── loader.py          # Data loading and train/test merging
    ├── features/
    │   └── engineering.py     # Feature engineering and sklearn pipeline
    ├── models/
    │   └── trainer.py         # Model training, CV evaluation, MLflow logging
    ├── visualisation/
    │   └── plots.py           # All plotting functions
    └── utils/
        ├── io.py              # File saving helpers
        └── logging.py         # Logging setup (with optional rich support)
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

## Setup and Installation

```bash
git clone <repo-url>
cd epfl-adsml-c3-02
pip install scikit-learn xgboost optuna mlflow pandas numpy matplotlib seaborn
```

`rich` is optional — logging falls back to plain stdout if it is not installed.

---

## How to Run

Open and run `notebooks/c3_p2_submission.ipynb` top to bottom.

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
