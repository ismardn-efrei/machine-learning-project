"""
FastAPI service exposing the trained stacking model for PlantVillage tabular features.

The API expects the *raw* numerical features (before scaling). They are normalized
using the saved StandardScaler and then fed to the stacking classifier. Both the
model and the scaler are stored under `artifacts/`.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - deployment guard
    raise RuntimeError("Pillow is required to run the API (pip install pillow).") from exc


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model_stacking.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.json"
LABEL_MAP_PATH = ARTIFACT_DIR / "label_map.json"


def _load_label_map(path: Path) -> Dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def _load_feature_columns(path: Path) -> List[str]:
    cols = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cols, list):
        raise ValueError("feature_columns.json must contain a list")
    return [str(c) for c in cols]


MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = _load_feature_columns(FEATURES_PATH)
LABEL_MAP = _load_label_map(LABEL_MAP_PATH)


def _image_stats(img: Image.Image) -> Dict[str, float]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    flat = arr.reshape(-1, 3)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    bright = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "brightness_mean": float(bright.mean()),
        "brightness_std": float(bright.std()),
    }


def _features_from_image_bytes(data: bytes) -> Dict[str, float]:
    if not data:
        raise ValueError("Empty file.")
    file_size_kb = round(len(data) / 1024.0, 3)
    with Image.open(BytesIO(data)) as img:
        w, h = img.size
        stats = _image_stats(img)
    aspect = round(w / h, 6) if h else 0.0
    features = {
        "file_size_kb": file_size_kb,
        "width": float(w),
        "height": float(h),
        "aspect_ratio": aspect,
    }
    features.update(stats)
    return features


class FeatureVector(BaseModel):
    file_size_kb: float = Field(..., description="Image file size in kilobytes")
    width: float
    height: float
    aspect_ratio: float
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    brightness_mean: float
    brightness_std: float

    def as_array(self) -> np.ndarray:
        values = [
            getattr(self, col)
            for col in FEATURE_COLUMNS
        ]
        return np.asarray(values, dtype=np.float32)[None, :]


class PredictionResponse(BaseModel):
    label_id: int
    label_name: str
    probabilities: Optional[Dict[str, float]] = None


app = FastAPI(title="Plant Disease Classifier API")


def _predict_from_vector(features: FeatureVector, top_k: int) -> PredictionResponse:
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    X = features.as_array()
    X_scaled = SCALER.transform(X)

    pred_id = int(MODEL.predict(X_scaled)[0])
    label_name = LABEL_MAP.get(pred_id, str(pred_id))

    proba: Optional[np.ndarray] = None
    if hasattr(MODEL, "predict_proba"):
        try:
            proba = MODEL.predict_proba(X_scaled)
        except Exception:
            proba = None

    top_probs: Optional[Dict[str, float]] = None
    if proba is not None:
        prob_vals = proba[0]
        pairs = sorted(
            ((LABEL_MAP.get(i, str(i)), float(p)) for i, p in enumerate(prob_vals)),
            key=lambda t: t[1],
            reverse=True,
        )[:top_k]
        top_probs = {name: prob for name, prob in pairs}

    return PredictionResponse(label_id=pred_id, label_name=label_name, probabilities=top_probs)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Plant Disease Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: radial-gradient(circle at top, #e0f2f1, #ffffff);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                margin: 0;
                padding: 40px 20px;
            }
            .card {
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 540px;
                padding: 30px;
            }
            h1 {
                margin-top: 0;
                text-align: center;
                color: #1b5e20;
            }
            form {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            input[type="file"] {
                padding: 12px;
                border: 2px dashed #80cbc4;
                border-radius: 10px;
                background: #f1f8e9;
            }
            button {
                border: none;
                border-radius: 8px;
                padding: 12px 16px;
                background: linear-gradient(120deg, #2e7d32, #43a047);
                color: #fff;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s ease, opacity 0.2s ease;
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            button:hover:not(:disabled) {
                transform: translateY(-1px);
            }
            #result {
                margin-top: 24px;
                padding: 18px;
                border-radius: 12px;
                background: #f4f9f4;
                border: 1px solid #c8e6c9;
                display: none;
            }
            #label {
                font-size: 1.1rem;
                color: #2e7d32;
                margin-bottom: 8px;
            }
            #probs {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            #probs li {
                margin: 4px 0;
            }
            .hint {
                margin-top: 18px;
                font-size: 0.9rem;
                color: #5f6a6a;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Plant Disease Classifier</h1>
            <form id="uploadForm">
                <input type="file" name="file" accept="image/*" required />
                <button type="submit" id="btnSubmit">Predict disease</button>
            </form>
            <div class="hint">Les résultats s’affichent ici. Vous pouvez aussi appeler l’API JSON sur <code>/predict</code>.</div>
            <div id="result">
                <div id="label"></div>
                <ul id="probs"></ul>
            </div>
        </div>

        <script>
        const form = document.getElementById("uploadForm");
        const btn = document.getElementById("btnSubmit");
        const resultBox = document.getElementById("result");
        const labelEl = document.getElementById("label");
        const probsEl = document.getElementById("probs");

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const fileInput = form.querySelector('input[type="file"]');
            if (!fileInput.files.length) {
                alert("Merci de sélectionner une image.");
                return;
            }
            const data = new FormData();
            data.append("file", fileInput.files[0]);
            btn.disabled = true;
            btn.textContent = "Analyse en cours...";
            resultBox.style.display = "none";

            try {
                const response = await fetch("/predict-image?top_k=3", {
                    method: "POST",
                    body: data
                });
                if (!response.ok) {
                    throw new Error(await response.text());
                }
                const json = await response.json();
                labelEl.textContent = `Classe prédite : ${json.label_name} (id: ${json.label_id})`;
                probsEl.innerHTML = "";
                if (json.probabilities) {
                    Object.entries(json.probabilities).forEach(([label, prob]) => {
                        const li = document.createElement("li");
                        li.textContent = `${label} — ${(prob * 100).toFixed(2)}%`;
                        probsEl.appendChild(li);
                    });
                }
                resultBox.style.display = "block";
            } catch (err) {
                alert("Erreur pendant la prédiction : " + err);
            } finally {
                btn.disabled = false;
                btn.textContent = "Predict disease";
            }
        });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: FeatureVector, top_k: int = 3) -> PredictionResponse:
    return _predict_from_vector(features, top_k)


@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(top_k: int = 3, file: UploadFile = File(...)) -> PredictionResponse:
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")
    try:
        feat_values = _features_from_image_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    fv = FeatureVector(**feat_values)
    return _predict_from_vector(fv, top_k)
