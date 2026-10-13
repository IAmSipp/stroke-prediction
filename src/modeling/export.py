import hashlib
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def export_browser_model():
    bundle_path = PROJECT_ROOT / "models" / "final_model_bundle.joblib"
    bundle = joblib.load(bundle_path)
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    columns = {name: list(names) for name, _, names in preprocessor.transformers_ if name != "remainder"}
    model = pipeline.named_steps["model"]
    exported = {
        "model_name": bundle["model_name"],
        "model_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "threshold": bundle["threshold"],
        "required_raw_features": bundle["required_raw_features"],
        "numeric": {
            "columns": columns["numeric"],
            "medians": numeric.named_steps["imputer"].statistics_.tolist(),
            "missing_indicators": numeric.named_steps["imputer"].indicator_.features_.tolist(),
            "mean": numeric.named_steps["scaler"].mean_.tolist(),
            "scale": numeric.named_steps["scaler"].scale_.tolist(),
        },
        "categorical": {
            "columns": columns["categorical"],
            "fill_values": categorical.named_steps["imputer"].statistics_.tolist(),
            "categories": [values.tolist() for values in categorical.named_steps["one_hot"].categories_],
        },
    }
    if isinstance(model, RandomForestClassifier):
        trees = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            values = tree.value[:, 0, :]
            scores = values[:, 1] / values.sum(axis=1)
            trees.append({
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "score": scores.tolist(),
            })
        exported["model"] = {"type": "random_forest", "trees": trees}
    elif isinstance(model, LogisticRegression):
        exported["model"] = {
            "type": "logistic_regression",
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }
    else:
        raise ValueError(f"Unsupported model: {type(model).__name__}")
    results_path = PROJECT_ROOT / "models" / "evaluation_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if results["model_sha256"] != exported["model_sha256"]:
            raise ValueError("Evaluate the current model before exporting it.")
        exported["test_metrics"] = next(
            row for row in results["metrics"]
            if row["model"] == "selected_model" and row["split"] == "test"
        )
    output_path = PROJECT_ROOT / "web" / "model.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(exported, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Exported {bundle['model_name']} to {output_path}")
    return exported


if __name__ == "__main__":
    export_browser_model()
