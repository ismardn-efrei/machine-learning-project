from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# 1. IMPORTATION DE MLFLOW (simplification de la logique)
try:
    import mlflow
except ImportError:
    mlflow = None
# -------------------------------------------------------------


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: Optional[float]


@dataclass
class ModelConfig:
    name: str
    hidden_dims: List[int]
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 30


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if len(class_names) <= 30:
        ticks = np.arange(len(class_names)) + 0.5
        plt.xticks(ticks, class_names, rotation=90)
        plt.yticks(ticks, class_names, rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_dataset(features_csv: Path, label_map_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(features_csv)
    label_col = "label_id"
    if label_col not in df.columns:
        raise ValueError("Expected a 'label_id' column in the features CSV.")
    feature_cols = [c for c in df.columns if c not in {"label_id", "label", "path"}]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    if label_map_path and label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        class_names = [label_map.get(str(i), str(i)) for i in range(int(np.max(y)) + 1)]
    else:
        class_names = [str(i) for i in range(int(np.max(y)) + 1)]
    return X, y, feature_cols, class_names


def split_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    val_ratio = 0.1 / 0.8  # 10% of the original data from train split
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=42, stratify=y_trainval
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return weights


def make_loaders_from_splits(
    splits: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    balance_strategy: str,
    class_weights: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    def data_loader(data_x: np.ndarray, data_y: np.ndarray, shuffle: bool = False, sampler=None) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler)

    sampler = None
    if balance_strategy in {"sampler", "both"} and class_weights is not None:
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = data_loader(X_train, y_train, shuffle=True, sampler=sampler)
    val_loader = data_loader(X_val, y_val, shuffle=False)
    test_loader = data_loader(X_test, y_test, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_labels.append(yb)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    return labels, preds, probs


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ModelMetrics:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = None
    try:
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
    )


def save_metrics(metrics: List[ModelMetrics], out_dir: Path) -> None:
    rows = [asdict(m) for m in metrics]
    df = pd.DataFrame(rows).sort_values(by=["f1_macro", "accuracy"], ascending=[False, False])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_summary.csv"
    json_path = out_dir / "metrics_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {csv_path}")


def compute_permutation_importance_nn(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    repeats: int = 5,
) -> np.ndarray:
    baseline_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    baseline_loader = DataLoader(baseline_dataset, batch_size=512, shuffle=False)
    y_base, preds_base, probs_base = evaluate_model(model, baseline_loader, device)
    baseline_metrics = compute_metrics("baseline", y_base, preds_base, probs_base)
    baseline_f1 = baseline_metrics.f1_macro
    rng = np.random.default_rng(42)
    drops = np.zeros(X_test.shape[1], dtype=float)

    for idx in range(X_test.shape[1]):
        drop_scores: List[float] = []
        for _ in range(repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, idx])
            perm_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_perm), torch.from_numpy(y_test)),
                batch_size=512,
                shuffle=False,
            )
            y_perm, preds_perm, probs_perm = evaluate_model(model, perm_loader, device)
            perm_metrics = compute_metrics("perm", y_perm, preds_perm, probs_perm)
            drop_scores.append(baseline_f1 - perm_metrics.f1_macro)
        drops[idx] = float(np.mean(drop_scores))
    return drops


def plot_feature_importance(importance: np.ndarray, feature_names: List[str], out_path: Path, title: str, top_n: int = 20) -> None:
    series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    if top_n:
        series = series.head(top_n)
    plt.figure(figsize=(8, max(4, int(0.4 * len(series)))))
    sns.barplot(x=series.values, y=series.index, color="#4C78A8")
    plt.title(title)
    plt.xlabel("F1 drop when shuffled")
    plt.ylabel("Feature")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    
def train_single_model(
    cfg: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    out_dir: Path,
    class_names: List[str],
    save_models: bool,
    mlflow_client, # Ici, c'est le drapeau 'mlflow_active' (booléen)
    class_weight_tensor: Optional[torch.Tensor] = None,
    feature_names: Optional[List[str]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    perm_importance: bool = False,
    perm_repeats: int = 5,
    perm_max_features: int = 64,
) -> ModelMetrics:
    
    mlflow_active = mlflow_client and mlflow is not None
        
    # MODIFICATION CLÉ 1: Démarrer un run imbriqué (nested=True)
    run_context = mlflow.start_run(run_name=cfg.name, nested=True) if mlflow_active else nullcontext()
    
    with run_context:
        if mlflow_active:
            mlflow.log_params(
                {
                    "model_name": cfg.name,
                    "hidden_dims": cfg.hidden_dims,
                    "dropout": cfg.dropout,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "batch_size": cfg.batch_size,
                    "epochs": cfg.epochs,
                }
            )
        model = MLP(input_dim, num_classes, cfg.hidden_dims, cfg.dropout).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_state = None
        best_val_f1 = -math.inf

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            avg_loss = epoch_loss / max(1, len(train_loader))

            y_val, preds_val, probs_val = evaluate_model(model, val_loader, device)
            val_metrics = compute_metrics(cfg.name, y_val, preds_val, probs_val)

            if mlflow_active:
                mlflow.log_metrics(
                    {
                        f"{cfg.name}_train_loss": avg_loss,
                        f"{cfg.name}_val_f1": val_metrics.f1_macro,
                    },
                    step=epoch,
                )

            if val_metrics.f1_macro > best_val_f1:
                best_val_f1 = val_metrics.f1_macro
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_test, preds_test, probs_test = evaluate_model(model, test_loader, device)
        metrics = compute_metrics(cfg.name, y_test, preds_test, probs_test)

        if mlflow_active:
            mlflow.log_metrics(
                {
                    f"{cfg.name}_accuracy": metrics.accuracy,
                    f"{cfg.name}_balanced_accuracy": metrics.balanced_accuracy,
                    f"{cfg.name}_precision_macro": metrics.precision_macro,
                    f"{cfg.name}_recall_macro": metrics.recall_macro,
                    f"{cfg.name}_f1_macro": metrics.f1_macro,
                }
            )
            if metrics.roc_auc_ovr is not None:
                mlflow.log_metric(f"{cfg.name}_roc_auc", metrics.roc_auc_ovr)
                
            # MODIFICATION CLÉ 2: Correction de l'appel mlflow.pytorch.log_model
            # Utilisation de 'artifact_path' et suppression de 'python_model'
            mlflow.pytorch.log_model(
                pytorch_model=model, 
                artifact_path=cfg.name, 
            )


        cm_path = out_dir / f"confusion_matrix_{cfg.name}.png"
        plot_confusion(y_test, preds_test, class_names, cm_path, f"Confusion Matrix - {cfg.name}")
        if mlflow_active:
            mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrices")

        if perm_importance and feature_names and X_test is not None and y_test is not None:
            if len(feature_names) > perm_max_features:
                print(f"[info] Skipping permutation importance for {cfg.name}: {len(feature_names)} features exceed limit ({perm_max_features}).")
            else:
                print(f"[info] Computing permutation importance for {cfg.name} ...")
                importances = compute_permutation_importance_nn(
                    model, X_test, y_test, device, repeats=perm_repeats
                )
                fi_path = out_dir / f"feature_importance_{cfg.name}.png"
                plot_feature_importance(importances, feature_names, fi_path, f"Feature Importance - {cfg.name}")
                if mlflow_active:
                    mlflow.log_artifact(str(fi_path), artifact_path="feature_importance")

        if save_models:
            model_path = out_dir / f"{cfg.name}.pt"
            torch.save(model.state_dict(), model_path)
            if mlflow_active:
                mlflow.log_artifact(str(model_path), artifact_path="models_pt") 

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train neural classifiers on PlantVillage features (GPU-ready)")
    parser.add_argument("--features-csv", type=str, required=True, help="Path to features CSV (scaled or embeddings).")
    parser.add_argument("--label-map", type=str, default="outputs/label_map.json", help="Path to label_map.json.")
    parser.add_argument("--out-dir", type=str, default="outputs/models", help="Directory to store metrics and artifacts.")
    parser.add_argument("--save-models", action="store_true", help="Save trained PyTorch models (.pt).")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging.")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI (e.g., file:./mlruns).")
    parser.add_argument("--mlflow-experiment", type=str, default="plant-disease-neural", help="MLflow experiment name.")
    parser.add_argument("--perm-importance", action="store_true", help="Compute permutation-based feature importance (slow).")
    parser.add_argument("--perm-repeats", type=int, default=5, help="Number of shuffles per feature for permutation importance.")
    parser.add_argument("--perm-max-features", type=int, default=64, help="Skip permutation importance if feature count exceeds this limit.")
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=["none", "class_weight", "sampler", "both"],
        default="none",
        help="Imbalance handling: use class-weighted loss, weighted sampler, or both.",
    )
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    label_map_path = Path(args.label_map) if args.label_map else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow_active = args.mlflow and mlflow is not None
    
    # 1. GESTION DU CONTEXTE MLFLOW GLOBAL
    run_context = nullcontext() # Par défaut, pas de contexte
    if mlflow_active:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        
        # Démarrage du RUN PARENT 
        run_context = mlflow.start_run(run_name="Project_Run_Summary") 
        mlflow_client = True # Drapau d'activité

    # Le reste du code est déplacé à l'intérieur du bloc 'with'
    with run_context: 
        X, y, feature_cols, class_names = load_dataset(features_csv, label_map_path)
        num_classes = len(class_names)
        input_dim = X.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Journalisation des paramètres globaux (sous le Run Parent)
        if mlflow_active:
            mlflow.log_param("features_csv", str(features_csv.resolve()))
            mlflow.log_param("label_map", str(label_map_path.resolve()) if label_map_path else "none")
            mlflow.log_param("samples", int(len(y)))
            mlflow.log_param("n_features", int(input_dim))
            mlflow.log_param("n_classes", int(num_classes))
            mlflow.log_param("balance_strategy", args.balance_strategy)
        
        configs = [
            ModelConfig(name="mlp_small", hidden_dims=[128, 64], dropout=0.2, lr=1e-3, epochs=30, batch_size=256),
            ModelConfig(name="mlp_medium", hidden_dims=[256, 128, 64], dropout=0.3, lr=1e-3, epochs=35, batch_size=256),
            ModelConfig(name="mlp_deep", hidden_dims=[512, 256, 128], dropout=0.4, lr=5e-4, epochs=40, batch_size=128),
        ]

        metrics_all: List[ModelMetrics] = []
        splits = split_dataset(X, y)
        X_train, y_train, X_val, y_val, X_test, y_test = splits
        class_weights_np = compute_class_weights(y_train, num_classes)
        use_class_weight = args.balance_strategy in {"class_weight", "both"}
        use_sampler = args.balance_strategy in {"sampler", "both"}
        sampler_weights = class_weights_np if use_sampler else None
        class_weight_tensor = (
            torch.tensor(class_weights_np, dtype=torch.float32, device=device) if use_class_weight else None
        )

        for cfg in configs:
            print(f"\n=== Training model: {cfg.name} ===")
            train_loader, val_loader, test_loader = make_loaders_from_splits(
                splits, cfg.batch_size, args.balance_strategy, sampler_weights
            )
            metrics = train_single_model(
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_dim=input_dim,
                num_classes=num_classes,
                device=device,
                out_dir=out_dir,
                class_names=class_names,
                save_models=args.save_models,
                mlflow_client=mlflow_active, # On passe le drapeau
                class_weight_tensor=class_weight_tensor,
                feature_names=feature_cols,
                X_test=X_test,
                y_test=y_test,
                perm_importance=args.perm_importance,
                perm_repeats=args.perm_repeats,
                perm_max_features=args.perm_max_features,
            )
            metrics_all.append(metrics)

        save_metrics(metrics_all, out_dir)
    # Le Run Parent se termine ici


if __name__ == "__main__":
    main()