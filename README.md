## Machine Learning Project Roadmap

### 1. Prétraitement + EDA
- Script: `main.py`
- Commande type: `./.venv/bin/python main.py --data-dir <PlantVillage>`  
  Produit `outputs/features_raw.csv`, `outputs/features_scaled.csv`, les figures EDA et `outputs/label_map.json`.

### 2. Entraînement et comparaison de modèles
- Script: `train.py`
- Exemple (échantillon 15k pour itérer rapidement) :
  ```
  ./.venv/bin/python train.py \
      --features-csv outputs/features_scaled_sample.csv \
      --label-map outputs/label_map.json \
      --out-dir outputs/models \
      --save-models --no-progress
  ```
- Résultats: `outputs/models/metrics_summary.{csv,json}`, matrices de confusion, modèles `.joblib`, graphes d’importance.
- Activer le suivi MLflow :
  ```
  ./.venv/bin/python train.py \
      --features-csv outputs/features_scaled.csv \
      --label-map outputs/label_map.json \
      --out-dir outputs/models \
      --save-models \
      --mlflow --mlflow-tracking-uri file:./mlruns \
      --mlflow-experiment plant-disease-tabular
  ```
  Puis lancer l’interface: `mlflow ui --backend-store-uri mlruns --port 5001`. Chaque run consigne les métriques, matrices de confusion, modèles sérialisés et figures d’importance.

### 3. Service FastAPI + Docker
- Le meilleur modèle (stacking) + scaler + label map sont copiés dans `artifacts/`.
- Lancer l’API localement :
  ```
  ./.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
  ```
- Requête exemple :
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "file_size_kb": 25.0,
      "width": 256,
      "height": 256,
      "aspect_ratio": 1.0,
      "mean_r": 130.0,
      "mean_g": 125.0,
      "mean_b": 110.0,
      "std_r": 52.0,
      "std_g": 45.0,
      "std_b": 55.0,
      "brightness_mean": 128.0,
      "brightness_std": 46.0
    }'
  ```
- Interface web : `http://localhost:8000/` expose un mini frontend (CSS + JS) pour uploader une image et afficher la classe prédite ainsi que les top probabilités.
- Upload direct via API (`/predict-image`) :
  ```
  curl -X POST "http://localhost:8000/predict-image?top_k=3" \
       -H "accept: application/json" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@/chemin/vers/image.jpg"
  ```
- Construction Docker :
  ```
  docker build -t plant-api .
  docker run -p 8000:8000 plant-api
  ```

### 4. Prochaines étapes
1. **Run complet MLflow** : relancer `train.py` sur la totalité du dataset (non échantillonné) avec `--mlflow` pour alimenter `mlruns/` et disposer de métriques finales suivies.
2. **Pipeline complet** : option pour recalculer les features + scaler + modèle finaux dans un même script (ou orchestrer via `make`/bash).
3. **Frontend / mini site** : bâtir un client (FastAPI templates / petit frontend) ou documenter les appels pour la démo.
4. **Rapport/présentation** : synthétiser l’EDA, la comparaison des modèles, l’interprétabilité, l’architecture API/Docker/MLflow, ainsi que le mode d’exécution.
