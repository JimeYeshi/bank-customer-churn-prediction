"""
Evaluate the saved churn prediction model and export reports/figures.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, roc_auc_score

from data_preprocessing import get_train_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
REPORT_PATH = PROJECT_ROOT / "outputs" / "classification_report.txt"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"


def main():
    _, X_test, _, y_test = get_train_test_data()
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("Classification Report\n")
        file.write("=====================\n\n")
        file.write(report)
        file.write(f"\nROC-AUC Score: {auc:.4f}\n")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Bank Customer Churn Model")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print(report)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Saved classification report to: {REPORT_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
