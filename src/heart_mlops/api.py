
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
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

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="5">
        <title>Heart Disease API Monitoring</title>
        <style>
            body {
                font-family: system-ui, sans-serif;
                max-width: 1000px;
                margin: 35px auto;
                padding: 0 20px;
                background: #f4f6f9;
                color: #172033;
            }
            h1 { margin-bottom: 6px; }
            .subtitle { color: #596579; margin-bottom: 24px; }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                gap: 16px;
            }
            .card {
                background: white;
                border-radius: 12px;
                padding: 22px;
                box-shadow: 0 2px 10px #0000000d;
            }
            .label { color: #596579; font-size: 14px; }
            .value { font-size: 32px; font-weight: 700; margin-top: 8px; }
            pre {
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                background: white;
                padding: 18px;
                border-radius: 12px;
            }
            a { color: #175cd3; }
            footer { margin-top: 24px; color: #596579; font-size: 13px; }
        </style>
    </head>
    <body>
        <h1>Heart Disease API Monitoring</h1>
        <p class="subtitle">
            Prometheus metrics · Auto-refreshes every 5 seconds
        </p>

        <div class="grid">
            <div class="card">
                <div class="label">Total HTTP requests</div>
                <div class="value" id="requests">Loading…</div>
            </div>
            <div class="card">
                <div class="label">Successful predictions</div>
                <div class="value" id="predictions">Loading…</div>
            </div>
            <div class="card">
                <div class="label">Prediction errors</div>
                <div class="value" id="errors">Loading…</div>
            </div>
            <div class="card">
                <div class="label">Average API latency</div>
                <div class="value" id="latency">Loading…</div>
            </div>
        </div>

        <h2>HTTP status breakdown</h2>
        <pre id="statuses">Loading metrics…</pre>

        <p>
            Raw metrics: <a href="/metrics">/metrics</a> ·
            API health: <a href="/health">/health</a> ·
            API documentation: <a href="/docs">/docs</a>
        </p>

        <footer>
            Educational project monitoring only. The model is not intended
            for clinical diagnosis.
        </footer>

        <script>
            async function refreshMetrics() {
                try {
                    const response = await fetch("/metrics", {cache: "no-store"});
                    if (!response.ok) throw new Error("Metrics endpoint unavailable");
                    const text = await response.text();
                    const lines = text.split("\\n");

                    function samples(prefix) {
                        return lines.filter(line =>
                            line.startsWith(prefix) && !line.startsWith("#")
                        );
                    }

                    function total(prefix) {
                        return samples(prefix).reduce((sum, line) => {
                            const match = line.match(/\\s+([0-9.eE+-]+)$/);
                            return sum + (match ? Number(match[1]) : 0);
                        }, 0);
                    }

                    const requestLines =
                        samples("heart_disease_http_requests_total");
                    const requestCount = requestLines.reduce((sum, line) => {
                        const match = line.match(/\\s+([0-9.eE+-]+)$/);
                        return sum + (match ? Number(match[1]) : 0);
                    }, 0);

                    const predictionLines =
                        samples("heart_disease_predictions_total");
                    const predictionSuccess = predictionLines
                        .filter(line => line.includes('outcome="success"'))
                        .reduce((sum, line) => {
                            const match = line.match(/\\s+([0-9.eE+-]+)$/);
                            return sum + (match ? Number(match[1]) : 0);
                        }, 0);
                    const predictionErrors = predictionLines
                        .filter(line => line.includes('outcome="error"'))
                        .reduce((sum, line) => {
                            const match = line.match(/\\s+([0-9.eE+-]+)$/);
                            return sum + (match ? Number(match[1]) : 0);
                        }, 0);

                    const countLines =
                        samples("heart_disease_http_request_duration_seconds_count");
                    const sumLines =
                        samples("heart_disease_http_request_duration_seconds_sum");
                    const count = countLines.reduce((sum, line) => {
                        const match = line.match(/\\s+([0-9.eE+-]+)$/);
                        return sum + (match ? Number(match[1]) : 0);
                    }, 0);
                    const duration = sumLines.reduce((sum, line) => {
                        const match = line.match(/\\s+([0-9.eE+-]+)$/);
                        return sum + (match ? Number(match[1]) : 0);
                    }, 0);

                    document.getElementById("requests").textContent =
                        requestCount.toLocaleString();
                    document.getElementById("predictions").textContent =
                        predictionSuccess.toLocaleString();
                    document.getElementById("errors").textContent =
                        predictionErrors.toLocaleString();
                    document.getElementById("latency").textContent =
                        count ? (duration / count * 1000).toFixed(2) + " ms" : "N/A";

                    const statuses = requestLines.map(line => {
                        const labels = line.match(/\\{([^}]*)\\}/);
                        const value = line.match(/\\s+([0-9.eE+-]+)$/);
                        return (labels ? labels[1] : "unknown") +
                            "  →  " + (value ? value[1] : "0");
                    });
                    document.getElementById("statuses").textContent =
                        statuses.length ? statuses.join("\\n") : "No requests recorded";
                } catch (error) {
                    document.getElementById("statuses").textContent =
                        "Unable to load metrics: " + error.message;
                }
            }

            refreshMetrics();
        </script>
    </body>
    </html>
    """

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
