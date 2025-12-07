"""
FastAPI service exposing a pre-trained PyTorch model for PlantVillage disease classification.
The API receives an image, extracts features using a ResNet18 backbone, scales them,
and then classifies them using a trained MLP model (e.g., 'mlp_deep.pt').
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from torch import nn
from torchvision import models, transforms

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required to run the API (pip install pillow).") from exc

# --- Configuration ---
ARTIFACTS_DIR = Path("artifacts")
CLASSIFIER_PATH = ARTIFACTS_DIR / "mlp_deep.pt"
LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map.json"

# --- Model Definition ---
class MLP(nn.Module):
    """Multi-Layer Perceptron for classification."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], dropout: float):
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

# --- Artifact Loading ---
def load_artifacts():
    """Loads all necessary models and preprocessing objects."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. ResNet model for feature extraction
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    embedding_model = models.resnet18(weights=weights)
    embedding_model.fc = torch.nn.Identity()
    embedding_model.eval()
    embedding_model.to(device)
    preprocess_tf = weights.transforms()

    # 2. MLP classifier model
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}")
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    num_classes = len(label_map)

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_PATH}")
    classifier_model = MLP(input_dim=512, num_classes=num_classes, hidden_dims=[512, 256, 128], dropout=0.4)
    classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier_model.eval()
    classifier_model.to(device)

    # 3. Label map
    int_label_map = {int(k): v for k, v in label_map.items()}

    return embedding_model, classifier_model, preprocess_tf, int_label_map, device

EMBEDDING_MODEL, CLASSIFIER_MODEL, PREPROCESS_TF, LABEL_MAP, DEVICE = load_artifacts()

# Static friendly labels for better display names
FRIENDLY_LABELS = {
    "Pepper__bell___Bacterial_spot": "Pepper (bell) - Bacterial spot",
    "Pepper__bell___healthy": "Pepper (bell) - Healthy",
    "Potato___Early_blight": "Potato - Early blight",
    "Potato___Late_blight": "Potato - Late blight",
    "Potato___healthy": "Potato - Healthy",
    "Tomato_Bacterial_spot": "Tomato - Bacterial spot",
    "Tomato_Early_blight": "Tomato - Early blight",
    "Tomato_Late_blight": "Tomato - Late blight",
    "Tomato_Leaf_Mold": "Tomato - Leaf mold",
    "Tomato_Septoria_leaf_spot": "Tomato - Septoria leaf spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato - Two-spotted spider mite",
    "Tomato__Target_Spot": "Tomato - Target spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato - Mosaic virus",
    "Tomato_healthy": "Tomato - Healthy",
}

def friendly_label(raw: str) -> str:
    """Converts an internal label name to a human-friendly one."""
    return FRIENDLY_LABELS.get(raw, raw.replace("__", " ").replace("_", " ").title())

def get_embedding_from_image_bytes(data: bytes) -> np.ndarray:
    """Extracts a ResNet feature vector from raw image bytes."""
    if not data:
        raise ValueError("Empty image file received.")
    try:
        with Image.open(BytesIO(data)) as img:
            img = img.convert("RGB")
            tensor = PREPROCESS_TF(img).unsqueeze(0).to(DEVICE)
    except Exception as exc:
        raise ValueError(f"Failed to process image: {exc}") from exc
        
    with torch.no_grad():
        feat = EMBEDDING_MODEL(tensor)
    
    return feat.squeeze().cpu().numpy()

# --- API Definition ---

class PredictionResponse(BaseModel):
    label_id: int
    label_name: str
    probabilities: Optional[Dict[str, float]] = None

app = FastAPI(title="Plant Disease Classifier API")

def _predict_from_embedding(embedding: np.ndarray, top_k: int) -> PredictionResponse:
    """Runs the prediction using the embedding."""
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    # Convert embedding to tensor
    X_tensor = torch.from_numpy(embedding.reshape(1, -1)).float().to(DEVICE)

    with torch.no_grad():
        logits = CLASSIFIER_MODEL(X_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze()

    pred_id = int(probabilities.argmax().item())
    label_name = friendly_label(LABEL_MAP.get(pred_id, str(pred_id)))

    # Get top_k probabilities
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(LABEL_MAP)))
    top_probs_dict = {
        friendly_label(LABEL_MAP.get(int(i), str(i))): float(p)
        for i, p in zip(top_indices, top_probs)
    }

    return PredictionResponse(label_id=pred_id, label_name=label_name, probabilities=top_probs_dict)


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
            <div class="hint">Results will appear below. You can also call the JSON API at <code>/predict-image</code>.</div>
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
                alert("Please select an image first.");
                return;
            }
            const data = new FormData();
            data.append("file", fileInput.files[0]);
            btn.disabled = true;
            btn.textContent = "Analyzing...";
            resultBox.style.display = "none";

            try {
                const response = await fetch("/predict-image?top_k=3", {
                    method: "POST",
                    body: data
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    let errorMessage = "Prediction failed";
                    try {
                        const errorJson = JSON.parse(errorText);
                        errorMessage = errorJson.detail || errorText;
                    } catch (e) {
                        errorMessage = errorText;
                    }
                    throw new Error(errorMessage);
                }
                const json = await response.json();
                labelEl.textContent = `Predicted class: ${json.label_name} (id: ${json.label_id})`;
                probsEl.innerHTML = "";
                if (json.probabilities) {
                    Object.entries(json.probabilities).forEach(([label, prob]) => {
                        const li = document.createElement("li");
                        li.textContent = `${label} : ${(prob * 100).toFixed(2)}%`;
                        probsEl.appendChild(li);
                    });
                }
                resultBox.style.display = "block";
            } catch (err) {
                alert(err.message);
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


@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(top_k: int = 3, file: UploadFile = File(...)) -> PredictionResponse:
    try:
        content = await file.read()
        embedding = get_embedding_from_image_bytes(content)
        return _predict_from_embedding(embedding, top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")