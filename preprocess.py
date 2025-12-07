"""
Data Preprocessing and EDA for the Plant Disease image dataset.

This script will:
  - Load the dataset folder (download via kagglehub if not provided).
  - Scan images and derive lightweight features per image (size, aspect, RGB
    channel stats, brightness, file size, etc.).
  - Handle missing/broken files robustly and encode labels.
  - Normalize/scale numerical features and export CSVs.
  - Produce EDA visualizations: class distribution, histograms/boxplots, and a
    correlation heatmap.

Outputs are saved under `outputs/` by default:
  - outputs/features_raw.csv
  - outputs/features_scaled.csv
  - outputs/label_map.json
  - outputs/scaler.joblib
  - outputs/feature_columns.json
  - outputs/eda/*.png

Usage examples:
  python preprocess.py --data-dir /path/to/PlantVillage
  python preprocess.py                      # attempts kagglehub download
  python preprocess.py --limit-per-class 200 --out-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps: Pillow, seaborn, scikit-learn, matplotlib
try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Pillow (PIL) is required. Please install with: pip install pillow"
    ) from e

try:  # matplotlib is common for visualizations
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required. Please install with: pip install matplotlib"
    ) from e

try:
    import seaborn as sns  # nicer visuals; optional fallback is handled
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required. Please install with: pip install scikit-learn"
    ) from e

try:
    import joblib

    _HAS_JOBLIB = True
except Exception:
    joblib = None
    _HAS_JOBLIB = False


def try_download_with_kagglehub() -> Optional[Path]:
    """Attempt to download the dataset via kagglehub.

    Returns the local path if successful; otherwise None.
    """
    try:
        import kagglehub  # type: ignore

        print("Downloading dataset with kagglehub: emmarex/plantdisease ...")
        path_str = kagglehub.dataset_download("emmarex/plantdisease")
        path = Path(path_str)
        print(f"Path to dataset files: {path}")
        return path
    except Exception as e:
        print("kagglehub download not available:", e)
        return None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_DIRS = {"train", "test", "val", "valid", "validation"}


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p


def infer_label_from_path(p: Path) -> str:
    """Infer label as the immediate parent directory name.

    If common split directories (train/test/val) are present, we take the
    parent above them.
    """
    parent = p.parent

    # If structure is .../split/class/img.jpg, prefer class
    if parent.name.lower() in SPLIT_DIRS and parent.parent.name:
        return parent.parent.name

    return parent.name


def image_stats(img: Image.Image) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return mean/std per RGB channel and brightness stats.

    Returns: mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness_mean, brightness_std
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    # shape: (H, W, 3)
    means = arr.reshape(-1, 3).mean(axis=0)
    stds = arr.reshape(-1, 3).std(axis=0)
    # Perceptual brightness (luma) in sRGB
    bright = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return (
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(stds[0]),
        float(stds[1]),
        float(stds[2]),
        float(bright.mean()),
        float(bright.std()),
    )


def scan_dataset(
    data_dir: Path,
    limit_per_class: Optional[int] = None,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """Scan dataset and compute per-image features.

    Features:
      - path, label, file_size_kb
      - width, height, aspect_ratio
      - mean/std RGB, brightness mean/std
    """
    rows = []
    class_counts: dict[str, int] = {}

    all_files = list(iter_image_files(data_dir))
    if max_images is not None:
        all_files = all_files[: max(0, int(max_images))]

    for fp in all_files:
        label = infer_label_from_path(fp)
        if limit_per_class is not None:
            n = class_counts.get(label, 0)
            if n >= limit_per_class:
                continue

        try:
            with Image.open(fp) as im:
                w, h = im.size
                means = image_stats(im)

            row = {
                "path": str(fp.resolve()),
                "label": label,
                "file_size_kb": round(fp.stat().st_size / 1024.0, 3),
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 6) if h else np.nan,
                "mean_r": means[0],
                "mean_g": means[1],
                "mean_b": means[2],
                "std_r": means[3],
                "std_g": means[4],
                "std_b": means[5],
                "brightness_mean": means[6],
                "brightness_std": means[7],
            }
            rows.append(row)
            class_counts[label] = class_counts.get(label, 0) + 1
        except Exception:
            # Skip unreadable/corrupt images
            continue

    df = pd.DataFrame(rows)
    return df


def eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution
    plt.figure(figsize=(12, 6))
    counts = df["label"].value_counts().sort_values(ascending=False)
    if _HAS_SEABORN:
        sns.barplot(x=counts.index, y=counts.values, color="#4C78A8")
    else:
        plt.bar(counts.index, counts.values, color="#4C78A8")
    plt.title("Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Histograms for numeric features
    num_cols = [
        "file_size_kb",
        "width",
        "height",
        "aspect_ratio",
        "mean_r",
        "mean_g",
        "mean_b",
        "brightness_mean",
    ]
    for col in num_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        if _HAS_SEABORN:
            sns.histplot(df[col].dropna(), kde=True, bins=40, color="#72B7B2")
        else:
            plt.hist(df[col].dropna(), bins=40, color="#72B7B2")
        plt.title(f"Histogram - {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()

    # Boxplots for selected features grouped by label (top classes)
    top_classes = df["label"].value_counts().head(6).index.tolist()
    df_top = df[df["label"].isin(top_classes)]
    box_cols = ["width", "height", "brightness_mean"]
    for col in box_cols:
        if col not in df_top.columns:
            continue
        plt.figure(figsize=(10, 5))
        if _HAS_SEABORN:
            sns.boxplot(data=df_top, x="label", y=col)
        else:
            # Fallback basic boxplot
            groups = [df_top[df_top["label"] == c][col].dropna().values for c in top_classes]
            plt.boxplot(groups, labels=top_classes)
        plt.title(f"Boxplot by Class - {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_{col}_by_label.png", dpi=150)
        plt.close()

    # Correlation heatmap among numeric features
    corr_cols = [
        "file_size_kb",
        "width",
        "height",
        "aspect_ratio",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "brightness_mean",
        "brightness_std",
    ]
    corr_df = df[corr_cols].select_dtypes(include=[np.number]).dropna()
    if len(corr_df) > 1:
        plt.figure(figsize=(10, 8))
        corr = corr_df.corr(numeric_only=True)
        if _HAS_SEABORN:
            sns.heatmap(corr, cmap="coolwarm", annot=False, square=True)
        else:
            plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png", dpi=150)
        plt.close()


def preprocess_and_export(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path, Path, Optional[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean: drop obvious NaNs
    df_clean = df.dropna().reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    df_clean["label_id"] = le.fit_transform(df_clean["label"].astype(str))
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}

    # Scale numeric features
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude label_id from scaling to keep an unscaled copy if needed
    to_scale = [c for c in numeric_cols if c not in {"label_id"}]
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df_clean[to_scale])
    df_scaled = df_clean.copy()
    df_scaled[to_scale] = scaled_vals

    # Save outputs
    raw_csv = out_dir / "features_raw.csv"
    scaled_csv = out_dir / "features_scaled.csv"
    map_json = out_dir / "label_map.json"
    df_clean.to_csv(raw_csv, index=False)
    df_scaled.to_csv(scaled_csv, index=False)
    map_json.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    scaler_path: Optional[Path]
    if _HAS_JOBLIB:
        scaler_path = out_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
    else:
        scaler_path = None
        print("Warning: joblib is unavailable; scaler will not be saved.")

    feature_cols_path = out_dir / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(to_scale, indent=2), encoding="utf-8")

    return raw_csv, scaled_csv, map_json, scaler_path, feature_cols_path


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing and EDA for Plant Disease images")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset root. If omitted, attempts kagglehub download.")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to write outputs (CSVs and plots).")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Limit number of images processed per class.")
    parser.add_argument("--max-images", type=int, default=None, help="Global maximum number of images to process.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    eda_dir = out_dir / "eda"

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"--data-dir not found: {data_dir}")
        print(f"Using dataset directory: {data_dir}")
    else:
        dl = try_download_with_kagglehub()
        if dl is None:
            raise RuntimeError("Dataset path not provided and kagglehub download failed.")
        data_dir = dl

    print("Scanning dataset and computing features ...")
    df = scan_dataset(
        data_dir=data_dir,
        limit_per_class=args.limit_per_class,
        max_images=args.max_images,
    )
    if df.empty:
        raise RuntimeError("No images found. Please verify dataset structure or provide --data-dir.")
    print(f"Scanned {len(df)} images across {df['label'].nunique()} classes.")

    print("Generating EDA plots ...")
    eda_plots(df, eda_dir)

    print("Preprocessing (encoding + scaling) and exporting CSVs ...")
    raw_csv, scaled_csv, map_json, scaler_path, feat_cols_path = preprocess_and_export(df, out_dir)

    print("Done.")
    print(f"Raw features: {raw_csv}")
    print(f"Scaled features: {scaled_csv}")
    print(f"Label map: {map_json}")
    if scaler_path is not None:
        print(f"Scaler: {scaler_path}")
    else:
        print("Scaler: <not saved - install joblib to persist the StandardScaler>")
    print(f"Feature columns: {feat_cols_path}")
    print(f"EDA figures: {eda_dir}")


if __name__ == "__main__":
    main()
