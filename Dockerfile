FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/best_model.joblib models/best_model.joblib

EXPOSE 8000

CMD ["uvicorn", "heart_mlops.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
