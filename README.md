# Plant Disease Classification API

This project classifies plant diseases from leaf images. It contains a REST API built with FastAPI to serve a trained model, along with scripts for the entire ML pipeline, from data processing to training.

The application is containerized using Docker for easy deployment and scalability.

## Quick Start with Docker

The recommended method for running the API is by using Docker.

**1. Build the Docker image:**
From the root directory of the project, run:
```bash
docker build -t plant-disease-api .
```

**2. Run the Docker container:**
This command starts the API, making it accessible at `http://localhost:8000`.
```bash
docker run -p 8000:8000 --name plant-api-container plant-disease-api
```
The API is now running and ready to accept requests.

---

## Usage

### API Request

Send a `POST` request with an image file to the `/predict-image` endpoint to get a classification.

Example using `curl`:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict-image?top_k=3' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/plant_leaf_image.jpg'
```
*Replace `/path/to/your/plant_leaf_image.jpg` with a valid file path.*

The server will return a JSON response:
```json
{
  "label_id": 1,
  "label_name": "Pepper (bell) - Healthy",
  "probabilities": {
    "Pepper (bell) - Healthy": 0.99,
    "Potato - Healthy": 0.005,
    "Tomato - Healthy": 0.005
  }
}
```

### Web Interface

A basic web interface for testing is available. Open your browser and navigate to `http://localhost:8000`.

### Interactive API Docs (Swagger UI)

For interactive testing, navigate to `http://localhost:8000/docs`. This interface allows you to upload images and send requests directly from your browser.

---

## Local Development Setup

If you prefer to run the application locally without Docker, follow these steps.

**1. Create a Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*(On Windows, use `venv\Scripts\activate`)*

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the API Server:**
```bash
uvicorn plant_api:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

---

## Model Training

The project includes scripts to run the full machine learning pipeline. The original dataset can be found in the `archive/` directory.

**1. Preprocess Data:**
This script processes the initial dataset and generates `outputs/features_raw.csv`.
```bash
python preprocess.py
```

**2. Extract Embeddings:**
This script uses a pre-trained ResNet18 model to extract image embeddings and saves them to `outputs/features_resnet18.csv`.
```bash
python extract_embeddings.py
```

**3. Train Models:**
This final script scales the features, trains the classifier, saves the final models to the `artifacts/` directory, and logs experiments with MLflow.
```bash
python train_models.py
```

---

## Project Structure

```
.
├── artifacts/              # Serialized models and artifacts for production
├── archive/                # Original raw dataset
├── mlruns/                 # MLflow experiment tracking data
├── outputs/                # Intermediate data files, EDA plots, and results
├── venv/                   # Python virtual environment
├── .gitignore
├── Dockerfile              # Docker configuration for the API
├── plant_api.py            # FastAPI application server
├── preprocess.py           # Script for initial data processing
├── extract_embeddings.py   # Script for feature extraction
├── train_models.py         # Script for model training
├── explain_model.py        # Script for model explainability
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
