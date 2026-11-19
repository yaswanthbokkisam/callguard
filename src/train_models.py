import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.modeling import LABELS, predict_dataframe, save_bundle, train_bundle


ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = ROOT / "data" / "synthetic_calls.csv"
VALIDATION_FILE = ROOT / "data" / "realistic_test.csv"
MODEL_FILE = ROOT / "models" / "callguard.joblib"
REPORT_FILE = ROOT / "reports" / "validation_metrics.json"


def calculate_metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }


def main():
    training_data = pd.read_csv(TRAINING_FILE)
    validation_data = pd.read_csv(VALIDATION_FILE)

    bundle = train_bundle(training_data)
    save_bundle(bundle, MODEL_FILE)

    predictions = predict_dataframe(
        bundle,
        validation_data["transcript"],
    )

    report = {
        "training_rows": len(training_data),
        "validation_rows": len(validation_data),
        "metrics": {},
    }

    for label in LABELS:
        report["metrics"][label] = calculate_metrics(
            validation_data[label],
            predictions[label],
        )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved report: {REPORT_FILE}")
    print("\nValidation metrics")

    metric_rows = []
    for label in LABELS:
        metrics = report["metrics"][label]
        metric_rows.append(
            {
                "label": label,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    print(
        pd.DataFrame(metric_rows).to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


if __name__ == "__main__":
    main()
