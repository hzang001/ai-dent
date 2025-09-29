#!/usr/bin/env python3
"""Train an oral cancer risk classifier on public biomarker-style data.

The script performs a stratified hold-out split and nested cross-validation
for model selection, then retrains the best estimator on the training split
and evaluates it on the hold-out set. All intermediate metrics and artefacts
are written to the ``artifacts`` directory by default.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_SOURCE_URL = "https://github.com/ahmedshaban26/oral-cancer"


@dataclass
class FoldResult:
    fold: int
    metrics: Dict[str, float]
    best_params: Dict[str, object]


@dataclass
class ModelEvaluation:
    name: str
    folds: List[FoldResult]

    @property
    def aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        metric_names = self.folds[0].metrics.keys()
        aggregated: Dict[str, Dict[str, float]] = {}
        for metric in metric_names:
            values = np.array([fold.metrics[metric] for fold in self.folds], dtype=float)
            aggregated[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        return aggregated

    @property
    def most_common_params(self) -> Dict[str, object]:
        param_counter: Counter = Counter()
        for fold in self.folds:
            items = tuple(sorted(fold.best_params.items()))
            param_counter[items] += 1
        best_items = next(iter(param_counter.most_common(1)))[0]
        return dict(best_items)

    @property
    def best_fold(self) -> FoldResult:
        return max(self.folds, key=lambda fr: fr.metrics["roc_auc"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/oral_cancer.csv"),
        help="Path to the oral cancer biomarker dataset (CSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where models and reports will be stored.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for the final hold-out evaluation.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    column_names = [
        "biomarker_1",
        "biomarker_2",
        "biomarker_3",
        "biomarker_4",
        "biomarker_5",
        "biomarker_6",
        "biomarker_7",
        "biomarker_8",
        "label",
    ]
    df = pd.read_csv(path, header=None, names=column_names)
    return df


def build_model_candidates() -> Dict[str, Dict[str, object]]:
    candidates: Dict[str, Dict[str, object]] = {
        "log_reg": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
        },
    }
    return candidates


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced),
    }


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_defs: Dict[str, Dict[str, object]],
    outer_splits: int = 5,
    inner_splits: int = 3,
) -> Dict[str, ModelEvaluation]:
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    evaluations: Dict[str, ModelEvaluation] = {
        name: ModelEvaluation(name=name, folds=[]) for name in model_defs
    }

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, cfg in model_defs.items():
            model = cfg["pipeline"]
            param_grid = cfg["param_grid"]
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            y_val_pred = search.predict(X_val)
            y_val_proba = search.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
            evaluations[name].folds.append(
                FoldResult(fold=fold_idx, metrics=metrics, best_params=search.best_params_)
            )
    return evaluations


def retrain_best_model(
    model_defs: Dict[str, Dict[str, object]],
    evaluations: Dict[str, ModelEvaluation],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[str, Pipeline]:
    best_model_name = max(
        evaluations.items(), key=lambda item: item[1].aggregated_metrics["roc_auc"]["mean"]
    )[0]
    best_eval = evaluations[best_model_name]

    # prefer the most common hyper-parameters across folds to reduce variance
    best_params = best_eval.most_common_params

    final_model: Pipeline = clone(model_defs[best_model_name]["pipeline"])
    final_model.set_params(**best_params)
    final_model.fit(X_train, y_train)
    return best_model_name, final_model


def evaluate_on_holdout(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def export_metrics(
    output_path: Path,
    dataset_summary: Dict[str, object],
    evaluations: Dict[str, ModelEvaluation],
    best_model_name: str,
    best_model_metrics: Dict[str, float],
) -> None:
    serialisable = {
        "dataset": dataset_summary,
        "model_selection": {
            name: {
                "cv_metrics": evaluation.aggregated_metrics,
                "folds": [
                    {
                        "fold": fold.fold,
                        "metrics": fold.metrics,
                        "best_params": fold.best_params,
                    }
                    for fold in evaluation.folds
                ],
            }
            for name, evaluation in evaluations.items()
        },
        "best_model": {
            "name": best_model_name,
            "holdout_metrics": best_model_metrics,
        },
        "data_source": DATA_SOURCE_URL,
    }
    output_path.write_text(json.dumps(serialisable, indent=2))


def export_feature_tables(
    reports_dir: Path,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values(by="permutation_importance_mean", ascending=False)
    perm_df.to_csv(reports_dir / "feature_permutation_importance.csv", index=False)

    model_step = model.named_steps.get("model")
    if hasattr(model_step, "coef_"):
        coef_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "coefficient": model_step.coef_[0],
            }
        ).sort_values(by="coefficient", key=np.abs, ascending=False)
        coef_df.to_csv(reports_dir / "feature_coefficients.csv", index=False)
    elif hasattr(model_step, "feature_importances_"):
        imp_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "feature_importance": model_step.feature_importances_,
            }
        ).sort_values(by="feature_importance", ascending=False)
        imp_df.to_csv(reports_dir / "feature_importances.csv", index=False)


def main() -> None:
    args = parse_args()
    df = load_dataset(args.data_path)

    feature_columns = [col for col in df.columns if col != "label"]
    X = df[feature_columns]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_definitions = build_model_candidates()
    evaluations = run_nested_cv(X_train, y_train, model_definitions)
    best_model_name, best_model = retrain_best_model(model_definitions, evaluations, X_train, y_train)

    holdout_metrics = evaluate_on_holdout(best_model, X_test, y_test)

    output_dir: Path = args.output_dir
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{best_model_name}_pipeline.joblib"
    dump(best_model, model_path)

    dataset_summary = {
        "n_samples": int(df.shape[0]),
        "n_features": int(len(feature_columns)),
        "class_distribution": {
            "positive": int(y.sum()),
            "negative": int((1 - y).sum()),
        },
        "feature_columns": feature_columns,
    }

    metrics_path = reports_dir / "model_performance.json"
    export_metrics(metrics_path, dataset_summary, evaluations, best_model_name, holdout_metrics)
    export_feature_tables(reports_dir, best_model, X_test, y_test)

    print(f"Best model: {best_model_name}")
    print(f"Hold-out ROC AUC: {holdout_metrics['roc_auc']:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
