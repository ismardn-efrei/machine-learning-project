"""
Model Training, Comparison, and Ensembling for the Plant Disease tabular features.

This script loads the precomputed features CSV (scaled) and trains multiple
classifiers, evaluates them on a held-out test set, and experiments with
ensemble methods (Voting and Stacking). Results and optional model artifacts
are saved under the specified output directory.

Usage examples:
  python train.py                                 # uses defaults under outputs/
  python train.py --features-csv outputs/features_scaled.csv \
                  --label-map outputs/label_map.json \
                  --out-dir outputs/models

Outputs (under --out-dir, default: outputs/models):
  - metrics_summary.csv         : Table of metrics per model
  - metrics_summary.json        : Same info in JSON
  - confusion_matrix_<model>.png: Confusion matrix heatmaps for each model
  - <model>.joblib              : Optional serialized models (enable with --save-models)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.inspection import permutation_importance

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: Optional[float]
    cv_f1_macro_mean: Optional[float]
    cv_f1_macro_std: Optional[float]


def _print_stage(title: str) -> None:
    print(f"\n[Stage] {title}")
    sys.stdout.flush()


class SimpleProgress:
    def __init__(self, total: int, label: str = "", width: int = 30, stream = sys.stdout):
        self.total = max(1, int(total))
        self.label = label
        self.width = max(10, int(width))
        self.stream = stream
        self.current = 0
        self._last_print_len = 0

    def _render(self, current: int) -> str:
        frac = min(1.0, max(0.0, current / self.total))
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(frac * 100)
        return f"{self.label} [{bar}] {current}/{self.total} ({pct}%)"

    def update(self, current: int) -> None:
        self.current = max(0, min(current, self.total))
        line = self._render(self.current)
        # Clear previous line if shorter/longer
        self.stream.write("\r" + line + " " * max(0, self._last_print_len - len(line)))
        self.stream.flush()
        self._last_print_len = len(line)

    def increment(self, step: int = 1) -> None:
        self.update(self.current + step)

    def done(self) -> None:
        self.update(self.total)
        self.stream.write("\n")
        self.stream.flush()


def load_dataset(features_csv: Path, label_col: str = "label_id") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(features_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {features_csv}")

    # X = numeric columns excluding the label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].astype(int).values
    return df, X, y, feature_cols


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> ModelMetrics:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    roc_auc = None
    if y_proba is not None:
        try:
            # Multiclass One-vs-Rest AUC if probabilities available
            roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            roc_auc = None

    return ModelMetrics(
        name=name,
        accuracy=float(acc),
        balanced_accuracy=float(bacc),
        precision_macro=float(prec),
        recall_macro=float(rec),
        f1_macro=float(f1),
        roc_auc_ovr=roc_auc,
        cv_f1_macro_mean=None,
        cv_f1_macro_std=None,
    )


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # Optionally limit tick labels if there are many classes
    if len(class_names) <= 30:
        plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=90)
        plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def attach_cv_scores(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    metrics: ModelMetrics,
    show_progress: bool = False,
) -> ModelMetrics:
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores: List[float] = []
        prog = SimpleProgress(cv.get_n_splits(), label=f"CV {name}") if show_progress else None
        for i, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            est = clone(estimator)
            est.fit(X[train_idx], y[train_idx])
            y_pred = est.predict(X[valid_idx])
            score = f1_score(y[valid_idx], y_pred, average="macro", zero_division=0)
            scores.append(score)
            if prog:
                prog.update(i)
        if prog:
            prog.done()
        metrics.cv_f1_macro_mean = float(np.mean(scores)) if len(scores) else None
        metrics.cv_f1_macro_std = float(np.std(scores)) if len(scores) else None
    except Exception:
        metrics.cv_f1_macro_mean = None
        metrics.cv_f1_macro_std = None
    return metrics


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    out_dir: Path,
    save_models: bool = False,
    show_progress: bool = True,
) -> Tuple[List[ModelMetrics], Dict[str, object], Tuple[np.ndarray, np.ndarray]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_stage("Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define base models
    models: Dict[str, object] = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=None),
        "svc_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=9),
        "gbc": GradientBoostingClassifier(random_state=42),
    }

    metrics_list: List[ModelMetrics] = []
    fitted: Dict[str, object] = {}

    _print_stage("Training Baseline Models")
    prog_models = SimpleProgress(len(models), label="Models") if show_progress else None
    for idx, (name, clf) in enumerate(models.items(), start=1):
        print(f"- Fitting: {name}")
        clf.fit(X_train, y_train)
        if show_progress and prog_models:
            prog_models.update(idx - 0.4)
        y_pred = clf.predict(X_test)

        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                y_proba = None
        elif hasattr(clf, "decision_function"):
            try:
                # Convert decision scores to pseudo-probabilities via softmax
                scores = clf.decision_function(X_test)
                if scores.ndim == 1:
                    scores = np.vstack([-scores, scores]).T
                e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                y_proba = e / np.sum(e, axis=1, keepdims=True)
            except Exception:
                y_proba = None

        m = compute_metrics(name, y_test, y_pred, y_proba)
        m = attach_cv_scores(name, clf, X, y, m, show_progress=show_progress)
        metrics_list.append(m)

        # Plot confusion matrix
        plot_confusion(y_test, y_pred, class_names, out_dir / f"confusion_matrix_{name}.png", f"Confusion Matrix - {name}")

        fitted[name] = clf
        # Optionally save model
        if save_models and _HAS_JOBLIB:
            joblib.dump(clf, out_dir / f"{name}.joblib")
        if show_progress and prog_models:
            prog_models.update(idx)
    if show_progress and prog_models:
        prog_models.done()

    # Ensembles: Voting (soft) and Stacking
    _print_stage("Ensemble: Voting (soft)")
    voting = VotingClassifier(
        estimators=[
            ("logreg", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
        ],
        voting="soft",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
    )
    if show_progress:
        pv = SimpleProgress(3, label="Voting fit")
    voting.fit(X_train, y_train)
    if show_progress:
        pv.done()
    y_pred = voting.predict(X_test)
    y_proba = None
    try:
        y_proba = voting.predict_proba(X_test)
    except Exception:
        y_proba = None
    mv = compute_metrics("voting_soft", y_test, y_pred, y_proba)
    mv = attach_cv_scores("voting_soft", voting, X, y, mv, show_progress=show_progress)
    metrics_list.append(mv)
    plot_confusion(y_test, y_pred, class_names, out_dir / "confusion_matrix_voting_soft.png", "Confusion Matrix - voting_soft")
    fitted["voting_soft"] = voting
    if save_models and _HAS_JOBLIB:
        joblib.dump(voting, out_dir / "voting_soft.joblib")

    _print_stage("Ensemble: Stacking")
    stacking = StackingClassifier(
        estimators=[
            ("logreg", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("svc", SVC(kernel="rbf", probability=True, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
        n_jobs=None,
    )
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_test)
    y_proba = None
    try:
        y_proba = stacking.predict_proba(X_test)
    except Exception:
        y_proba = None
    ms = compute_metrics("stacking", y_test, y_pred, y_proba)
    ms = attach_cv_scores("stacking", stacking, X, y, ms, show_progress=show_progress)
    metrics_list.append(ms)
    plot_confusion(y_test, y_pred, class_names, out_dir / "confusion_matrix_stacking.png", "Confusion Matrix - stacking")
    fitted["stacking"] = stacking
    if save_models and _HAS_JOBLIB:
        joblib.dump(stacking, out_dir / "stacking.joblib")

    # Save a classification report for best model (by f1_macro)
    best = max(metrics_list, key=lambda r: r.f1_macro)
    print(f"Best model by f1_macro: {best.name} ({best.f1_macro:.4f})")

    return metrics_list, fitted, (X_test, y_test)


def resolve_class_names(label_map_path: Optional[Path], y: np.ndarray) -> List[str]:
    n_classes = int(np.max(y)) + 1
    names = [str(i) for i in range(n_classes)]
    if label_map_path and label_map_path.exists():
        try:
            m = json.loads(label_map_path.read_text(encoding="utf-8"))
            # keys may be strings representing ints
            names = [m.get(str(i), names[i]) for i in range(n_classes)]
        except Exception:
            pass
    return names


def save_metrics(metrics: List[ModelMetrics], out_dir: Path) -> None:
    rows = [asdict(m) for m in metrics]
    df = pd.DataFrame(rows)
    # Sort by f1 macro descending
    df = df.sort_values(by=["f1_macro", "accuracy"], ascending=[False, False])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {out_dir / 'metrics_summary.csv'}")
    return None


def extract_builtin_importance(model, feature_names: List[str]) -> Optional[pd.Series]:
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            if imp.shape[0] == len(feature_names):
                return pd.Series(imp, index=feature_names).sort_values(ascending=False)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 1:
                coef = coef[None, :]
            imp = np.mean(np.abs(coef), axis=0)
            if imp.shape[0] == len(feature_names):
                return pd.Series(imp, index=feature_names).sort_values(ascending=False)
    except Exception:
        return None
    return None


def compute_permutation_importance(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], n_repeats: int = 8, random_state: int = 42, scoring: str = "f1_macro") -> Optional[pd.Series]:
    try:
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
        imp = pd.Series(r.importances_mean, index=feature_names)
        return imp.sort_values(ascending=False)
    except Exception:
        return None


def plot_feature_importance(imp: pd.Series, out_path: Path, title: str, top_n: int = 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top = imp.head(top_n)[::-1]
    plt.figure(figsize=(8, max(5, int(0.35 * len(top)))))
    sns.barplot(x=top.values, y=top.index, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train, compare, and ensemble models on tabular image features")
    parser.add_argument("--features-csv", type=str, default="outputs/features_scaled.csv", help="Path to scaled features CSV.")
    parser.add_argument("--label-map", type=str, default="outputs/label_map.json", help="Path to label_map.json (optional).")
    parser.add_argument("--out-dir", type=str, default="outputs/models", help="Directory to write metrics and artifacts.")
    parser.add_argument("--save-models", action="store_true", help="Serialize trained models with joblib.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    parser.add_argument("--perm-importance", action="store_true", help="Force permutation importance for all models (in addition to built-in where available).")
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    label_map_path = Path(args.label_map) if args.label_map else None
    out_dir = Path(args.out_dir)

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    print(f"Loading dataset from: {features_csv}")
    df, X, y, feature_cols = load_dataset(features_csv)
    class_names = resolve_class_names(label_map_path, y)

    print(f"Samples: {len(y)} | Features: {len(feature_cols)} | Classes: {len(class_names)}")
    print("Training models and evaluating...")
    metrics, models, (X_test, y_test) = train_and_evaluate(
        X, y, class_names, out_dir=out_dir, save_models=args.save_models, show_progress=not args.no_progress
    )
    save_metrics(metrics, out_dir)

    # Interpretability: compute and save feature importance plots (no report)
    _print_stage("Interpretability: Feature Importance")
    df_rows = [asdict(m) for m in metrics]
    dfm = pd.DataFrame(df_rows).sort_values(by=["f1_macro", "accuracy"], ascending=[False, False])
    candidates: List[str] = []
    if not dfm.empty:
        candidates.append(dfm.iloc[0]["name"])  # best model
    for ens in ["voting_soft", "stacking"]:
        if ens in models:
            candidates.append(ens)

    seen: set[str] = set()
    for name in candidates:
        if name in seen or name not in models:
            continue
        seen.add(name)
        model = models[name]
        title = f"Feature Importance - {name}"
        out_path = out_dir / f"feature_importance_{name}.png"
        imp = extract_builtin_importance(model, feature_cols)
        if imp is None or args.perm_importance:
            # fallback to permutation on held-out test set
            imp = compute_permutation_importance(model, X_test, y_test, feature_cols)
        if imp is not None:
            plot_feature_importance(imp, out_path, title)
    print("Done.")


if __name__ == "__main__":
    main()
