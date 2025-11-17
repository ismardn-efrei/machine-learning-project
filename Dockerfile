FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY plant_api.py plant_api.py
COPY artifacts artifacts

EXPOSE 8000

CMD ["uvicorn", "plant_api:app", "--host", "0.0.0.0", "--port", "8000"]
