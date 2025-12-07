
import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split
from torch import nn

# This MLP class is copied from train_models.py to ensure the architecture matches
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

def main():
    parser = argparse.ArgumentParser(description="Explain a trained MLP model using SHAP.")
    parser.add_argument("--model-path", type=str, default="outputs/models/mlp_deep.pt", help="Path to the trained .pt model file.")
    parser.add_argument("--config-name", type=str, default="mlp_deep", help="Name of the model config to use (mlp_small, mlp_medium, mlp_deep).")
    parser.add_argument("--features-csv", type=str, default="outputs/features_resnet18.csv", help="Path to the features CSV used for training.")
    parser.add_argument("--label-map", type=str, default="artifacts/label_map.json", help="Path to label_map.json.")
    parser.add_argument("--output-path", type=str, default="outputs/models/shap_summary.png", help="Path to save the SHAP summary plot.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to explain. Use a small number as KernelExplainer is slow.")
    parser.add_argument("--background-samples", type=int, default=100, help="Number of background samples for the explainer.")

    args = parser.parse_args()

    # --- 1. Load Data ---
    print("Loading data...")
    features_path = Path(args.features_csv)
    label_map_path = Path(args.label_map)

    df = pd.read_csv(features_path)
    # In features_resnet18.csv, columns are '0', '1', ... '511', 'label_id'
    feature_cols = [c for c in df.columns if c not in {"label_id", "label", "path"}]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_id"].values.astype(int)

    with open(label_map_path) as f:
        label_map = json.load(f)
    class_names = [label_map.get(str(i), str(i)) for i in range(max(y) + 1)]
    num_classes = len(class_names)
    input_dim = X.shape[1]

    # Use the same splits as in training to get a representative test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Load Model ---
    print(f"Loading model from {args.model_path}...")
    
    # Recreate the model with the same architecture as during training
    model_configs = {
        "mlp_small": {"hidden_dims": [128, 64], "dropout": 0.2},
        "mlp_medium": {"hidden_dims": [256, 128, 64], "dropout": 0.3},
        "mlp_deep": {"hidden_dims": [512, 256, 128], "dropout": 0.4},
    }
    
    config = model_configs[args.config_name]
    model = MLP(input_dim=input_dim, num_classes=num_classes, hidden_dims=config['hidden_dims'], dropout=config['dropout'])
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # --- 3. Set up SHAP Explainer ---
    # SHAP's KernelExplainer needs a function that takes a numpy array (n_samples, n_features)
    # and returns a numpy array (n_samples, n_classes) of probabilities.
    def predict_proba(numpy_array):
        tensor = torch.from_numpy(numpy_array).float()
        with torch.no_grad():
            logits = model(tensor)
            probas = torch.softmax(logits, dim=1)
        return probas.numpy()

    # To make KernelExplainer faster, we summarize the background data using a small sample.
    print(f"Creating SHAP explainer with {args.background_samples} background samples...")
    background_data = shap.sample(X_train, args.background_samples)
    explainer = shap.KernelExplainer(predict_proba, background_data)

    # --- 4. Calculate SHAP values ---
    # We'll explain a subset of the test data to save time.
    samples_to_explain_df = pd.DataFrame(X_test).sample(args.num_samples, random_state=42)
    samples_to_explain = samples_to_explain_df.values
    
    print(f"Calculating SHAP values for {args.num_samples} samples... (This may take a while)")
    shap_values = explainer.shap_values(samples_to_explain)

    # --- 5. Generate and Save Plot ---
    print(f"Generating and saving SHAP summary plot to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # The summary plot can handle the list of arrays for multi-class problems directly.
    plt.figure()
    shap.summary_plot(
        shap_values,
        samples_to_explain,
        feature_names=feature_cols,
        class_names=class_names,
        show=False,
        plot_size=None, # Let shap decide the plot size automatically
        max_display=20 # Display top 20 features
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"SHAP summary plot saved successfully to {args.output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
