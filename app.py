import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
)

matplotlib.use("Agg")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PLOTS_FOLDER = "static/plots"
ALLOWED_EXTENSIONS = {"csv"}
MODEL_DIR = "model"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

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


def load_artifacts():
    arts = {}
    for name, fname in [
        ("lgbm", "lgbm_model.pkl"),
        ("xgb", "xgb_model.pkl"),
        ("scaler", "scaler.pkl"),
        ("diagnostics", "diagnostics.pkl"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            arts[name] = joblib.load(path)
    return arts


ARTIFACTS = load_artifacts()


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


def predict_ensemble(X_scaled):
    lgbm = ARTIFACTS.get("lgbm")
    xgb = ARTIFACTS.get("xgb")
    probas = []
    if lgbm:
        probas.append(lgbm.predict_proba(X_scaled)[:, 1])
    if xgb:
        probas.append(xgb.predict_proba(X_scaled)[:, 1])
    if not probas:
        return None, None
    avg = np.stack(probas).mean(axis=0)
    return (avg >= 0.5).astype(int), avg


def shap_top_features(X_row_scaled, n=3):
    lgbm = ARTIFACTS.get("lgbm")
    if lgbm is None:
        return []
    explainer = shap.TreeExplainer(lgbm)
    sv = explainer.shap_values(X_row_scaled)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()
    pairs = sorted(zip(FEATURE_COLS, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if not file or not allowed_file(file.filename):
            return "Invalid file. Upload a CSV.", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            required = {
                "type", "amount",
                "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest",
            }
            missing = required - set(df.columns)
            if missing:
                return f"CSV missing columns: {missing}", 400

            feature_df = engineer_features(df)
            scaler = ARTIFACTS.get("scaler")
            X = scaler.transform(feature_df.values) if scaler else feature_df.values

            preds, probas = predict_ensemble(X)
            if preds is None:
                return "Models not loaded. Run train.py first.", 500

            ctx = {
                "fraud_count": int(preds.sum()),
                "total_count": len(preds),
            }

            if "isFraud" in df.columns:
                y_true = df["isFraud"].values
                ctx["accuracy"] = round(float(accuracy_score(y_true, preds)), 6)
                ctx["precision"] = round(float(precision_score(y_true, preds, zero_division=0)), 6)
                ctx["recall"] = round(float(recall_score(y_true, preds, zero_division=0)), 6)
                ctx["f1"] = round(float(f1_score(y_true, preds, zero_division=0)), 6)
                ctx["roc_auc"] = round(float(roc_auc_score(y_true, probas)), 6)

                fpr, tpr, _ = roc_curve(y_true, probas)
                plt.figure()
                plt.plot(fpr, tpr, color="blue", lw=2,
                         label=f"AUC = {ctx['roc_auc']:.4f}")
                plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="lower right")
                roc_path = os.path.join(PLOTS_FOLDER, "roc_curve.png")
                plt.savefig(roc_path)
                plt.close()

                prec_v, rec_v, _ = precision_recall_curve(y_true, probas)
                plt.figure()
                plt.plot(rec_v, prec_v, color="purple", lw=2,
                         label=f"AP = {ctx['precision']:.4f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend(loc="lower left")
                pr_path = os.path.join(PLOTS_FOLDER, "precision_recall_curve.png")
                plt.savefig(pr_path)
                plt.close()

                ctx["roc_plot_url"] = url_for("static", filename="plots/roc_curve.png")
                ctx["pr_plot_url"] = url_for("static", filename="plots/precision_recall_curve.png")

            return render_template("index.html", **ctx)

        except Exception as e:
            return f"Error processing file: {str(e)}", 500

    return render_template("index.html")


@app.route("/predict_transaction", methods=["GET", "POST"])
def predict_transaction():
    if request.method == "POST":
        try:
            txn_type = request.form.get("transaction_type")
            amount = float(request.form.get("amount", 0))
            old_orig = float(request.form.get("oldbalanceOrg", 0))
            new_orig = float(request.form.get("newbalanceOrig", 0))
            old_dest = float(request.form.get("oldbalanceDest", 0))
            new_dest = float(request.form.get("newbalanceDest", 0))

            row = pd.DataFrame([{
                "type": txn_type, "amount": amount,
                "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
                "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            }])

            feature_df = engineer_features(row)
            scaler = ARTIFACTS.get("scaler")
            X = scaler.transform(feature_df.values) if scaler else feature_df.values

            preds, probas = predict_ensemble(X)
            if preds is None:
                return "Models not loaded. Run train.py first.", 500

            is_fraud = "Fraud" if preds[0] == 1 else "Not Fraud"
            confidence = round(float(probas[0]), 4)

            top = shap_top_features(X)
            top_str = ", ".join(f"{n} ({v:+.3f})" for n, v in top) if top else "N/A"

            notes = []
            if amount > 10000:
                notes.append("High transaction amount.")
            bc_orig = (new_orig - old_orig) / (old_orig + 1)
            if abs(bc_orig) > 0.5:
                notes.append("Large originator balance change.")
            if old_orig > 0 and new_orig == 0:
                notes.append("Originator balance fully drained.")
            if old_dest == new_dest and amount > 0:
                notes.append("Destination balance unchanged despite transaction.")
            notes.append(f"Top SHAP features: {top_str}.")
            explanation = " ".join(notes)

            return render_template(
                "predict_transaction.html",
                transaction_type=txn_type,
                is_fraud=is_fraud,
                confidence_score=confidence,
                explanation=explanation,
            )

        except Exception as e:
            return f"Error: {str(e)}", 500

    return render_template("transaction_form.html")


@app.route("/transactions", methods=["GET"])
def transactions():
    return {"message": "No live transaction feed configured."}, 200


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    feedback_path = os.path.join(UPLOAD_FOLDER, "feedback.csv")
    row = pd.DataFrame([data])
    row.to_csv(feedback_path, mode="a", header=not os.path.exists(feedback_path), index=False)
    return {"status": "recorded"}, 200


if __name__ == "__main__":
    app.run(debug=True)
