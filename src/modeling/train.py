import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.features.selection import LogicalFeatureSelector
from src.features.stroke_features import StrokeFeatureEngineer, get_required_raw_features
from src.modeling.pipeline import build_model_pipeline, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
TARGET_RECALL = 0.80


def choose_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
    })
    eligible = table[table["recall"] >= TARGET_RECALL]
    chosen = eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]
    return float(chosen["threshold"]), table


def train_models():
    data_dir = PROJECT_ROOT / "data" / "processed"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv")
    validation = pd.read_csv(data_dir / "validation.csv")
    X_train, y_train = split_features_target(train)
    X_validation, y_validation = split_features_target(validation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    specifications = {
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
            {
                "feature_selection__k": [8, 10, "all"],
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_features": ["sqrt", 1.0],
            },
        ),
    }
    searches = {}
    rows = []
    for name, (estimator, parameters) in specifications.items():
        print(f"Training {name}...", flush=True)
        use_engineering = name != "logistic_regression"
        pipeline = build_model_pipeline(
            estimator,
            feature_engineer=StrokeFeatureEngineer() if use_engineering else None,
            feature_selector=LogicalFeatureSelector(random_state=RANDOM_STATE) if use_engineering else None,
        )
        search = RandomizedSearchCV(
            pipeline,
            parameters,
            n_iter=5 if name == "logistic_regression" else 12,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            refit="pr_auc",
            cv=cv,
            n_jobs=2,
            random_state=RANDOM_STATE,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[name] = search
        index = search.best_index_
        rows.append({
            "model": name,
            "cv_pr_auc": float(search.best_score_),
            "cv_pr_auc_std": float(search.cv_results_["std_test_pr_auc"][index]),
            "cv_train_pr_auc": float(search.cv_results_["mean_train_pr_auc"][index]),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][index]),
            "best_parameters": search.best_params_,
        })
        print(f"CV PR-AUC: {search.best_score_:.4f}", flush=True)

    comparison = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    selected_name = comparison.iloc[0]["model"]
    selected = searches[selected_name]
    pipeline = selected.best_estimator_
    validation_probability = pipeline.predict_proba(X_validation)[:, 1]
    threshold, threshold_table = choose_threshold(y_validation, validation_probability)
    selector = pipeline.named_steps.get("feature_selection")
    selected_features = selector.selected_features_.tolist() if selector else X_train.columns.tolist()
    required_raw_features = get_required_raw_features(selected_features, X_train.columns)
    feature_scores = []
    if selector is not None:
        feature_scores = pd.DataFrame({
            "feature": selector.feature_names_in_,
            "mi_score": selector.scores_,
            "selected": selector.get_support(),
        }).sort_values("mi_score", ascending=False).to_dict(orient="records")

    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_name": selected_name,
        "selected_logical_features": selected_features,
        "required_raw_features": required_raw_features,
    }
    joblib.dump(bundle, model_dir / "final_model_bundle.joblib", compress=3)
    metadata = {
        "model_name": selected_name,
        "task": "binary_classification",
        "target": "stroke",
        "labels": {"0": "no stroke", "1": "stroke"},
        "selection_metric": "average_precision (PR-AUC)",
        "train_cv_pr_auc": float(selected.best_score_),
        "train_cv_pr_auc_std": float(comparison.iloc[0]["cv_pr_auc_std"]),
        "validation_pr_auc": float(average_precision_score(y_validation, validation_probability)),
        "validation_roc_auc": float(roc_auc_score(y_validation, validation_probability)),
        "threshold": threshold,
        "threshold_reason": "highest validation precision with recall >= 80%",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "refit_on_validation": False,
        "selected_logical_features": selected_features,
        "required_raw_features": required_raw_features,
        "best_parameters": selected.best_params_,
        "sklearn_version": sklearn.__version__,
        "split_sha256": {
            name: hashlib.sha256((data_dir / f"{name}.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in ("train", "validation")
        },
    }
    manifest = {
        "selection_method": "mutual_information_inside_cv" if selector else "all_original_features",
        "selected_feature_count": len(selected_features),
        "selected_logical_features": selected_features,
        "required_raw_features": required_raw_features,
        "feature_scores": feature_scores,
    }
    for name, content in [("model_metadata.json", metadata), ("feature_manifest.json", manifest)]:
        (model_dir / name).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    comparison.to_csv(model_dir / "model_comparison.csv", index=False)
    threshold_table.to_csv(model_dir / "validation_thresholds.csv", index=False)
    print(f"Selected {selected_name}; threshold={threshold:.4f}", flush=True)
    return comparison, metadata, threshold_table


if __name__ == "__main__":
    comparison, metadata, _ = train_models()
    print(comparison.to_string(index=False))
