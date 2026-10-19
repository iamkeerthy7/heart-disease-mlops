# Heart Disease Classification - MLOps Assignment

## 1. Project overview

This project implements an end-to-end machine learning workflow for binary
classification using the UCI Cleveland Heart Disease dataset. It demonstrates
data acquisition, exploratory data analysis (EDA), preprocessing, model
selection, experiment tracking, automated testing, containerisation, API
serving, Kubernetes deployment, and basic monitoring.

The models compared are:
- Logistic Regression
- Random Forest

The selected model is determined by mean cross-validation ROC-AUC, rather
than by selecting a model using the test set.

**Educational-use disclaimer:** This project is not intended for clinical
diagnosis, treatment decisions, or other medical decision-making.

## 2. Target definition and dataset

The original target column is `num`. It is converted to a binary target:

- `0`: `num = 0` — absence of diagnosed heart disease in the dataset.
- `1`: `num > 0` — presence of diagnosed heart disease in the dataset.

The dataset contains 303 records, 13 input features, and the original target.
The source data contain missing values in `ca` and `thal`, which are handled
by the preprocessing pipeline.

The data acquisition script is located at `src/heart_mlops/data.py`.
Dataset provenance and additional notes are documented in `data/README.md`.

Do not interpret the target as a clinical diagnosis for a real person.

## 3. Repository structure

```text
.
├── data/
│   ├── heart.csv
│   └── README.md
├── src/heart_mlops/
│   ├── __init__.py
│   ├── data.py
│   ├── eda.py
│   ├── train.py
│   └── api.py
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── reports/
│   ├── figures/
│   ├── missing_values.csv
│   └── model_comparison.csv
├── tests/
├── deployment/k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated files such as MLflow databases, run artifacts, Python caches, and
the virtual environment may also exist locally.

## 4. Prerequisites

For local development:

- Python 3.12 is recommended to match the CI environment.
- Git
- Internet access for initial dependency installation and data acquisition,
  if the dataset is not already present.

For container execution, install Docker.

For the Kubernetes workflow, install `kubectl` and Minikube. The Kubernetes
commands below assume a local Minikube cluster and the manifests in
`deployment/k8s/`.

## 5. Local environment setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/iamkeerthy7/heart-disease-mlops.git
cd heart-disease-mlops
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make the source package importable for the commands below:

```bash
export PYTHONPATH="$PWD/src"
```

Run subsequent Python commands from the repository root with the virtual
environment activated.

## 6. Data acquisition

To run the data acquisition script:

```bash
python -m heart_mlops.data
```

The script is responsible for obtaining the dataset when required. If
`data/heart.csv` already exists, inspect the script's behaviour before
re-downloading or replacing the file.

Check that the dataset is present:

```bash
ls -lh data/heart.csv
```

## 7. Exploratory data analysis

Generate the EDA outputs:

```bash
python -m heart_mlops.eda
```

The analysis produces figures and a missing-value summary under `reports/`,
including:

- Original and binary target distributions
- Feature histograms
- Categorical feature distributions
- Feature correlation heatmap
- Missing-value summary

Inspect the generated outputs:

```bash
find reports -maxdepth 2 -type f | sort
```

## 8. Model training and evaluation

Run the training workflow:

```bash
python -m heart_mlops.train
```

The workflow performs preprocessing, compares Logistic Regression and
Random Forest, tunes model hyperparameters with stratified cross-validation,
evaluates the selected candidates on a held-out test set, logs experiments
to MLflow, and saves the selected reusable model and metadata.

The preprocessing pipeline includes numerical imputation and scaling, plus
categorical imputation and one-hot encoding. Keeping preprocessing within
the fitted pipeline helps ensure consistent transformations during inference.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Cross-validation results

Generated outputs include the saved model in `models/`, model metadata,
the model-comparison CSV, confusion-matrix plots, and MLflow run artifacts.

Review the comparison table:

```bash
cat reports/model_comparison.csv
```

The saved model is intended for educational demonstration. Performance
measured on this dataset does not establish clinical validity or suitability
for real-world use.

## 9. MLflow experiment tracking

Training records model parameters, evaluation metrics, plots, and model
artifacts in MLflow. The exact tracking location depends on the environment
and the tracking configuration.

For the local SQLite-backed tracking setup used by the project, start the
MLflow UI from the repository root:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000` in a browser on the same machine.

If the UI is running on a remote lab machine, use the lab's approved port
forwarding or browser-access mechanism. Do not expose the tracking UI
publicly without appropriate access controls.

If the database or run artifacts are stored elsewhere, use the tracking URI
configured for that environment. Avoid deleting the tracking database or
artifact directories if you need to retain experiment history.

## 10. Automated tests and code quality

Run the test suite:

```bash
pytest -v tests
```

Run lint checks, if Ruff is installed:

```bash
ruff check src tests
```

Check Python syntax by compiling the source and test files:

```bash
python -m compileall -q src tests
```

The tests cover API behaviour and preprocessing/model-related behaviour.
The GitHub Actions workflow in `.github/workflows/ci.yml` automates code
quality checks, tests, and model training. Check the Actions page in GitHub
for the result of each workflow run.

## 11. Run the API locally

Start the FastAPI application from the repository root:

```bash
export PYTHONPATH="$PWD/src"
uvicorn heart_mlops.api:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | Basic service information |
| `/health` | Health and model-loading status |
| `/predict` | Make a prediction using a JSON request |
| `/metrics` | Prometheus-format metrics |
| `/dashboard` | Basic monitoring dashboard |

Check service health:

```bash
curl -s http://127.0.0.1:8000/health
```

Example prediction request:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 3
  }'
```

Use this payload only to demonstrate the request format; it is not a real
patient assessment. The API response includes the prediction and associated
confidence/probability information supported by the implementation.

Check monitoring endpoints:

```bash
curl -s http://127.0.0.1:8000/metrics
```

Open `http://127.0.0.1:8000/dashboard` in a browser to view the dashboard.

## 12. Build and run with Docker

Build the image from the repository root:

```bash
docker build -t heart-disease-api:1.0 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 heart-disease-api:1.0
```

In another terminal, verify health:

```bash
curl -s http://127.0.0.1:8000/health
```

Test `/predict`, `/metrics`, and `/dashboard` using the local API instructions
above. Stop the container with `Ctrl+C` in the terminal where it is running.

The API must be tested against the same image that was built. If the source
code changes, rebuild the image before testing the updated container.

## 13. Kubernetes deployment with Minikube

Start a local Minikube cluster if one is not already running:

```bash
minikube start
minikube addons enable ingress
```

Build the image in the Docker environment used by Minikube. With a Docker
driver, a common approach is:

```bash
eval "$(minikube docker-env)"
docker build -t heart-disease-api:1.0 .
```

Apply the deployment, service, and ingress manifests:

```bash
kubectl apply -f deployment/k8s/
kubectl rollout status deployment/heart-disease-api
kubectl get pods
kubectl get services
kubectl get ingress
```

The deployment image name and Kubernetes resource names must match the
manifests in `deployment/k8s/`. If the image is not available to the cluster,
check the Minikube image environment and the image pull policy.

For an Ingress-based local route, obtain the Minikube IP:

```bash
minikube ip
```

Check the Ingress address and NGINX controller status:

```bash
kubectl describe ingress
kubectl get pods -n ingress-nginx
```

If the Ingress host in `deployment/k8s/ingress.yaml` is a hostname rather
than a catch-all rule, send the corresponding `Host` header in requests.
For example, replace `<INGRESS_HOST>` with the host configured in the manifest:

```bash
curl -H 'Host: <INGRESS_HOST>' http://$(minikube ip)/health
```

Test `/predict`, `/metrics`, and `/dashboard` through the working local route.
The exact URL depends on the Ingress configuration and your lab environment.

A Minikube address is a local cluster address, not a public internet endpoint.
A Kubernetes `LoadBalancer` service may show `EXTERNAL-IP` as `pending` in
Minikube; this alone does not mean the service is broken. Follow the
Ingress/local-access instructions for the configured environment.

To inspect or troubleshoot the deployment:

```bash
kubectl get all
kubectl logs deployment/heart-disease-api
kubectl describe deployment heart-disease-api
```

## 14. Request logging and monitoring

The API records request information such as the method, path, response
status, and request duration. The Prometheus-format `/metrics` endpoint
exposes application metrics, including HTTP request counts, request
durations, and prediction outcomes as implemented by the service.

The `/dashboard` endpoint provides a simple view of selected operational
metrics. Generate a few requests and refresh the dashboard to observe
changes.

These features are a basic educational monitoring implementation. They do
not replace production monitoring, alerting, access control, audit
requirements, or privacy review.

## 15. Reproducibility and troubleshooting

For a reproducible run:

1. Use the documented Python version and install `requirements.txt`.
2. Run commands from the repository root.
3. Set `PYTHONPATH="$PWD/src"` when running modules or the local API.
4. Confirm the dataset exists before EDA and training.
5. Review the training output, saved model, comparison CSV, and MLflow runs.
6. Run tests and lint checks before building the container.
7. Rebuild the Docker image after source changes.
8. Confirm the Kubernetes cluster can access the image and that the rollout
   succeeds before testing endpoints.

If a command fails, retain the complete error output and check the relevant
configuration before changing dependencies or deleting generated artifacts.

## 16. Limitations and responsible use

This project is an academic MLOps demonstration, not a clinically validated
medical system. Its dataset is limited in size and scope. Its evaluation
metrics do not establish generalisation to other populations, calibration,
fairness, clinical utility, or safety. The API should not be used to make
medical decisions.

The local Minikube deployment is intended for demonstration and does not
automatically provide a publicly accessible production endpoint. A public
deployment, if required, needs an appropriately configured cloud or hosting
environment and security controls.

## 17. Submission evidence

Before submission, capture genuine evidence of:

- EDA figures and missing-value analysis
- Model comparison and evaluation metrics
- MLflow runs, parameters, metrics, and artifacts
- Successful tests and the latest GitHub Actions run
- Docker build and a successful API request
- Kubernetes deployment status and a working endpoint
- API metrics, request logging, and the monitoring dashboard

Include the screenshots in the report and the repository's screenshot
directory. Also include the final report and a short screen-recorded
demonstration as required by the assignment instructions.
