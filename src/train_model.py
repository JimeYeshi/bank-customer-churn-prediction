"""
Train and compare machine learning models for customer churn prediction.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, get_train_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.csv"


def evaluate_predictions(y_true, y_pred, y_proba):
    """Return standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    X_train, X_test, y_train, y_test = get_train_test_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    best_model = None
    best_model_name = None
    best_auc = -1

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        results.append(metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = pipeline
            best_model_name = model_name

    metrics_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

    print("Model training completed.")
    print(f"Best model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(metrics_df)


if __name__ == "__main__":
    main()
