# Financial Fraud Detection

Built a time-aware, leakage-controlled ML pipeline on the PaySim synthetic mobile-money dataset. Automated feature synthesis produces eight transaction-level signals (balance-change ratios, drain flags, balance-error terms, amount-to-balance ratio) on top of one-hot-encoded transaction types. Class imbalance is handled with SMOTE + RandomUnderSampler applied to training folds only. A LightGBM and XGBoost ensemble is tuned via Bayesian optimization (Optuna, 50 trials each) using rolling-window validation built on `TimeSeriesSplit` over the `step` column, keeping future data strictly out of training. SHAP `TreeExplainer` surfaces per-feature attributions and an error-slicing pass identifies segments where the ensemble makes disproportionate mistakes. Final ensemble val metrics: accuracy 99.99%, ROC-AUC 0.9967.

## Features

- **Batch Processing:** Upload a PaySim-format CSV and get fraud predictions with accuracy, precision, recall, F1, ROC-AUC, and ROC/PR curve plots (when ground-truth labels are present).
- **Single Transaction Prediction:** Submit one transaction via form and receive a fraud verdict, confidence score, and SHAP-based explanation of the top driving features.
- **SHAP Explanations:** Each prediction surfaces the top features pushing the model toward or away from fraud.
- **Analyst Feedback Loop:** POST feedback on incorrect predictions to `/feedback` for future retraining.
- **Summary Statistics:** Fraud count and total transaction count returned on every batch upload.

## Tech Stack

- **Backend:** Flask, Flask-CORS, Pandas, NumPy, Joblib
- **Models:** LightGBM, XGBoost (ensemble)
- **Tuning:** Optuna (Bayesian optimization, TPE sampler)
- **Explainability:** SHAP (TreeExplainer)
- **Imbalance handling:** imbalanced-learn (SMOTE, RandomUnderSampler)
- **Validation:** scikit-learn TimeSeriesSplit
- **Visualization:** Matplotlib

## Project Structure

```
├── train.py               # Full training pipeline (run this first)
├── app.py                 # Flask web app
├── templates/
│   ├── index.html         # Batch upload + results
│   ├── transaction_form.html
│   └── predict_transaction.html
├── model/                 # Artifacts saved by train.py
│   ├── lgbm_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── diagnostics.pkl    # SHAP importances, error slices, best params
├── uploads/               # Place the PaySim CSV here
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- PaySim dataset CSV placed at `uploads/PS_20174392719_1491204439457_log.csv`

### Installation

```bash
git clone https://github.com/Danielmark001/MLDA-finfraud.git
cd MLDA-finfraud
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Loads the first 100K rows, engineers features, runs 5-fold rolling-window splits, tunes LightGBM and XGBoost with Optuna, computes SHAP values, runs error slicing, and saves all artifacts to `model/`.

### Run the app

```bash
python app.py
```

Starts the Flask server at `http://localhost:5000`.

## CSV Format

Batch upload expects standard PaySim columns:

| Column | Description |
|---|---|
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Originator opening balance |
| `newbalanceOrig` | Originator closing balance |
| `oldbalanceDest` | Destination opening balance |
| `newbalanceDest` | Destination closing balance |
| `isFraud` | (optional) Ground-truth label — enables metric computation |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET/POST | `/` | Batch CSV upload and results |
| GET/POST | `/predict_transaction` | Single transaction prediction form |
| GET | `/transactions` | Transaction feed placeholder |
| POST | `/feedback` | Submit analyst correction |

## Dataset Source

Axi, E. (2018). Synthetic Financial Datasets For Fraud Detection. Kaggle.
https://www.kaggle.com/datasets/ealaxi/paysim1/data
