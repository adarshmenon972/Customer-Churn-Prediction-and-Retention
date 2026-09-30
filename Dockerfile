FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# Train at build time so the image ships with a model (swap for your own data/artifact in production)
RUN python -m churnguard.train --n-samples 20000 --n-trials 15 --out artifacts
ENV CHURN_MODEL_PATH=/app/artifacts/churn_model.joblib
EXPOSE 8000
CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
