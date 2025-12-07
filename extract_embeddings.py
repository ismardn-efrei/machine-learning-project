"""
Embedding extraction pipeline for PlantVillage images using a pretrained ResNet.

This script walks the dataset directory, loads each image, feeds it through a
ResNet18 backbone pretrained on ImageNet, and stores the resulting feature
vector alongside the inferred label. The final CSV can be used as input to
train_models.py (instead of the lightweight tabular features).

Usage example:
  python extract_embeddings.py \
      --data-dir /path/to/PlantVillage \
      --out-csv outputs/features_resnet18.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_DIRS = {"train", "test", "val", "valid", "validation"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p


def infer_label_from_path(p: Path) -> str:
    parent = p.parent
    if parent.name.lower() in SPLIT_DIRS and parent.parent.name:
        return parent.parent.name
    return parent.name


def load_model(device: torch.device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


def extract_feature(
    path: Path,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze().cpu().numpy()


def run(
    data_dir: Path,
    out_csv: Path,
    limit_per_class: Optional[int],
    max_images: Optional[int],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess_tf = load_model(device)

    rows: List[np.ndarray] = []
    labels: List[str] = []
    paths: List[str] = []
    class_counts: dict[str, int] = {}

    files = list(iter_image_files(data_dir))
    if max_images is not None:
        files = files[: max(0, int(max_images))]

    total = len(files)
    print(f"Found {total} candidate images.")
    for idx, path in enumerate(files, start=1):
        label = infer_label_from_path(path)
        if limit_per_class is not None:
            n = class_counts.get(label, 0)
            if n >= limit_per_class:
                continue
        try:
            emb = extract_feature(path, model, preprocess_tf, device)
        except Exception as exc:
            print(f"[warn] failed on {path}: {exc}")
            continue

        rows.append(emb)
        labels.append(label)
        paths.append(str(path.resolve()))
        class_counts[label] = class_counts.get(label, 0) + 1

        if idx % 250 == 0:
            print(f"Processed {idx}/{total} images ...")

    if not rows:
        raise RuntimeError("No embeddings extracted; check dataset path.")

    feat_dim = rows[0].shape[0]
    columns = [f"emb_{i}" for i in range(feat_dim)]
    df_features = pd.DataFrame(rows, columns=columns)
    df_features["label"] = labels
    df_features["path"] = paths

    le = LabelEncoder()
    df_features["label_id"] = le.fit_transform(labels)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_csv, index=False)
    print(f"Saved embeddings to {out_csv} ({len(df_features)} samples, dim={feat_dim}).")

    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    map_path = out_csv.parent / "label_map.json"
    map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    print(f"Label map refreshed at {map_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ResNet embeddings for PlantVillage dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder.")
    parser.add_argument("--out-csv", type=str, default="outputs/features_resnet18.csv", help="CSV path for embeddings.")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional maximum images per class.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional global maximum number of images.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    run(
        data_dir=data_dir,
        out_csv=Path(args.out_csv),
        limit_per_class=args.limit_per_class,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
