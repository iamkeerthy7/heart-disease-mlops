
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

HTTP_REQUESTS = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "heart_disease_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)
PREDICTION_REQUESTS = Counter(
    "heart_disease_predictions_total",
    "Prediction requests by outcome",
    ["outcome"],
)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

app = FastAPI(
    title="Heart Disease Classification API",
    description=(
        "Educational ML service using the UCI Cleveland Heart Disease dataset. "
        "Not intended for clinical diagnosis."
    ),
    version="1.0.0",
)

model = None


class HeartDiseaseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        endpoint = request.url.path

        # Avoid high-cardinality labels such as query strings or input values.
        HTTP_REQUESTS.labels(
            request.method, endpoint, str(status_code)
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            request.method, endpoint
        ).observe(duration)

        logger.info(
            "method=%s path=%s status=%s duration_seconds=%.6f",
            request.method,
            endpoint,
            status_code,
            duration,
        )


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run the training script first."
        )

    model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Classification API",
        "docs": "/docs",
        "metrics": "/metrics",
        "disclaimer": "Educational use only; not for clinical diagnosis.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
def predict(payload: HeartDiseaseInput):
    if model is None:
        PREDICTION_REQUESTS.labels(outcome="model_unavailable").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([payload.model_dump()], columns=FEATURES)

    try:
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))

        PREDICTION_REQUESTS.labels(outcome="success").inc()

        return {
            "prediction": prediction,
            "prediction_label": (
                "positive" if prediction == 1 else "negative"
            ),
            "confidence": round(confidence, 4),
            "probabilities": {
                str(int(label)): round(float(probability), 4)
                for label, probability in zip(model.classes_, probabilities)
            },
            "disclaimer": "Educational output only; not a medical diagnosis.",
        }
    except Exception as exc:
        PREDICTION_REQUESTS.labels(outcome="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input values and model.",
        ) from exc
