from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
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
        "disclaimer": "Educational use only; not for clinical diagnosis.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(payload: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([payload.model_dump()], columns=FEATURES)

    try:
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))

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
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the input values and model.",
        ) from exc
