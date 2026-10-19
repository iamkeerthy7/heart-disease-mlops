# Heart Disease Classification — MLOps Assignment

## Overview

This educational project builds a binary classifier using the UCI Cleveland
Heart Disease dataset. It compares Logistic Regression and Random Forest,
tracks experiments with MLflow, serves predictions through FastAPI, and
packages the API in Docker.

**Disclaimer:** This project is for educational purposes only. It is not
intended for clinical diagnosis or medical decision-making.

## Target definition

The original target is `num`. The binary target is defined as:
- `0`: num = 0
- `1`: num > 0

## Selected model

The training script selects a model using mean cross-validation ROC-AUC.
The current selected model is Logistic Regression.

## Project layout

- `data/`: dataset and data-source notes
- `src/heart_mlops/data.py`: dataset acquisition
- `src/heart_mlops/eda.py`: exploratory data analysis
- `src/heart_mlops/train.py`: model training, tuning, and MLflow tracking
- `src/heart_mlops/api.py`: FastAPI prediction service
- `models/`: saved model and metadata
- `reports/`: evaluation results and plots
- `tests/`: automated API tests
- `.github/workflows/ci.yml`: GitHub Actions CI workflow
- `Dockerfile`: container image definition
- `deployment/kubernetes/`: Kubernetes deployment files

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
