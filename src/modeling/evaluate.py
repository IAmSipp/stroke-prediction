import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modeling.pipeline import build_model_pipeline, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_model():
    model_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data" / "processed"
    bundle_path = model_dir / "final_model_bundle.joblib"
    bundle = joblib.load(bundle_path)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    for name in ("train", "validation"):
        checksum = hashlib.sha256((data_dir / f"{name}.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if checksum != metadata["split_sha256"][name]:
            raise ValueError("The data splits changed. Train the model again before evaluation.")
    model_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    split_hashes = {
        name: hashlib.sha256((data_dir / f"{name}.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in ("train", "validation", "test")
    }
    result_path = model_dir / "evaluation_results.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved["model_sha256"] == model_hash and saved["split_sha256"] == split_hashes:
            print("Loaded the saved evaluation for this model and these splits.")
            return pd.DataFrame(saved["metrics"]), saved

    splits = {name: pd.read_csv(data_dir / f"{name}.csv") for name in split_hashes}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(splits[left]["id"]) & set(splits[right]["id"]):
            raise ValueError(f"Patient IDs overlap between {left} and {right}.")
    X_train, y_train = split_features_target(splits["train"])
    baselines = {
        "dummy_prior": build_model_pipeline(DummyClassifier(strategy="prior")),
        "baseline_logistic": build_model_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
        ),
    }
    for pipeline in baselines.values():
        pipeline.fit(X_train, y_train)
    models = {"selected_model": (bundle["pipeline"], bundle["threshold"])}
    models.update({name: (pipeline, 0.5) for name, pipeline in baselines.items()})
    rows = []
    for model_name, (pipeline, threshold) in models.items():
        for split_name, frame in splits.items():
            features, labels = split_features_target(frame)
            if model_name == "selected_model":
                features = features[bundle["required_raw_features"]]
            probabilities = pipeline.predict_proba(features)[:, 1]
            rows.append({
                "model": model_name,
                "split": split_name,
                "rows": len(frame),
                "positive_count": int(labels.sum()),
                "threshold": threshold,
                **calculate_metrics(labels, probabilities, threshold),
            })
            if model_name == "selected_model" and split_name == "test":
                pd.DataFrame({
                    "id": frame["id"],
                    "stroke": labels,
                    "model_score": probabilities,
                    "prediction": (probabilities >= threshold).astype(int),
                }).to_csv(model_dir / "test_predictions.csv", index=False)
    results = pd.DataFrame(rows)
    saved = {
        "model_name": bundle["model_name"],
        "model_sha256": model_hash,
        "split_sha256": split_hashes,
        "threshold_source": "validation",
        "test_used_for_tuning": False,
        "metrics": rows,
    }
    results.to_csv(model_dir / "evaluation_metrics.csv", index=False)
    result_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    return results, saved


if __name__ == "__main__":
    results, _ = evaluate_model()
    print(results.to_string(index=False))
