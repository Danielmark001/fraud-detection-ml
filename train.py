import os
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50
DATA_PATH = os.path.join("uploads", "PS_20174392719_1491204439457_log.csv")
MODEL_DIR = "model"

FEATURE_COLS = [
    "amount",
    "balance_change_orig",
    "balance_change_dest",
    "orig_balance_zeroed",
    "dest_balance_unchanged",
    "amount_to_orig_balance",
    "orig_balance_error",
    "dest_balance_error",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def engineer_features(df):
    df = df.copy()

    df["balance_change_orig"] = (
        (df["newbalanceOrig"] - df["oldbalanceOrg"]) / (df["oldbalanceOrg"] + 1)
    )
    df["balance_change_dest"] = (
        (df["newbalanceDest"] - df["oldbalanceDest"]) / (df["oldbalanceDest"] + 1)
    )
    df["orig_balance_zeroed"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    df["dest_balance_unchanged"] = (
        df["oldbalanceDest"] == df["newbalanceDest"]
    ).astype(int)
    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs()
    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()

    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if col not in type_dummies.columns:
            type_dummies[col] = 0

    df = pd.concat([df, type_dummies], axis=1)
    return df[FEATURE_COLS]


def rolling_window_splits(df, n_splits=N_SPLITS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sorted_positions = df["step"].argsort().values
    for train_pos, val_pos in tscv.split(sorted_positions):
        yield sorted_positions[train_pos], sorted_positions[val_pos]


def tune_lgbm(X_tr, y_tr, X_val, y_val):
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "scale_pos_weight": pos_weight,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def tune_xgb(X_tr, y_tr, X_val, y_val):
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "scale_pos_weight": pos_weight,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        m = xgb.XGBClassifier(**params, verbosity=0)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50,
        )
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def error_slicing(model, X_val, y_val, threshold=0.5):
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    errors = (preds != y_val).astype(int)

    X_df = pd.DataFrame(X_val, columns=FEATURE_COLS)
    slices = []
    for col in FEATURE_COLS:
        unique_vals = X_df[col].unique()
        if len(unique_vals) <= 5:
            for val in sorted(unique_vals):
                mask = X_df[col] == val
                if mask.sum() >= 50:
                    slices.append({
                        "feature": col,
                        "value": float(val),
                        "n": int(mask.sum()),
                        "error_rate": round(float(errors[mask].mean()), 4),
                    })

    return sorted(slices, key=lambda x: x["error_rate"], reverse=True)[:10]


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, nrows=100_000)
    print(f"  {len(df)} rows loaded")

    feature_df = engineer_features(df)
    X = feature_df.values
    y = df["isFraud"].values

    splits = list(rolling_window_splits(df, n_splits=N_SPLITS))
    print(f"Rolling-window splits: {N_SPLITS} folds on step-sorted data")

    # use last fold for tuning and final eval
    train_idx, val_idx = splits[-1]
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    print(f"  train: {len(X_tr)} rows  |  val: {len(X_val)} rows")

    # imbalance handling on training fold only
    smote = SMOTE(random_state=RANDOM_STATE)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    X_tr_res, y_tr_res = rus.fit_resample(X_tr_res, y_tr_res)
    print(f"  after SMOTE+RUS: {len(X_tr_res)} rows  (fraud={y_tr_res.sum()})")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_res)
    X_val_scaled = scaler.transform(X_val)

    print(f"\nTuning LightGBM ({N_TRIALS} trials) ...")
    lgbm_params = tune_lgbm(X_tr_scaled, y_tr_res, X_val_scaled, y_val)
    lgbm_params.update({
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": (y_tr_res == 0).sum() / max((y_tr_res == 1).sum(), 1),
    })
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_tr_scaled, y_tr_res)
    print("  LightGBM best params:", lgbm_params)

    print(f"\nTuning XGBoost ({N_TRIALS} trials) ...")
    xgb_params = tune_xgb(X_tr_scaled, y_tr_res, X_val_scaled, y_val)
    xgb_params.update({
        "objective": "binary:logistic", "eval_metric": "auc",
        "random_state": RANDOM_STATE, "verbosity": 0,
        "scale_pos_weight": (y_tr_res == 0).sum() / max((y_tr_res == 1).sum(), 1),
    })
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr_scaled, y_tr_res)
    print("  XGBoost best params:", xgb_params)

    # SHAP on a sample to keep it fast
    print("\nComputing SHAP values ...")
    sample_size = min(500, len(X_val_scaled))
    explainer = shap.TreeExplainer(lgbm_model)
    sv = explainer.shap_values(X_val_scaled[:sample_size])
    if isinstance(sv, list):
        sv = sv[1]
    shap_importance = {
        name: round(float(np.abs(sv[:, i]).mean()), 6)
        for i, name in enumerate(FEATURE_COLS)
    }
    print("  SHAP mean |value| per feature:")
    for k, v in sorted(shap_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"    {k}: {v}")

    # error slicing
    print("\nRunning error slicing ...")
    slices = error_slicing(lgbm_model, X_val_scaled, y_val)
    if slices:
        print("  top error slices:")
        for s in slices[:5]:
            print(f"    {s['feature']}={s['value']}  n={s['n']}  err={s['error_rate']}")

    # ensemble metrics
    lgbm_prob = lgbm_model.predict_proba(X_val_scaled)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_val_scaled)[:, 1]
    ensemble_prob = 0.5 * lgbm_prob + 0.5 * xgb_prob
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, ensemble_pred)), 6),
        "precision": round(float(precision_score(y_val, ensemble_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_val, ensemble_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_val, ensemble_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_val, ensemble_prob)), 6),
    }
    print("\nEnsemble metrics:", metrics)

    # save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm_model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(
        {
            "shap_importance": shap_importance,
            "error_slices": slices,
            "metrics": metrics,
            "lgbm_params": lgbm_params,
            "xgb_params": xgb_params,
            "feature_cols": FEATURE_COLS,
        },
        os.path.join(MODEL_DIR, "diagnostics.pkl"),
    )
    print(f"\nAll artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
