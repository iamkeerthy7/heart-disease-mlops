from pathlib import Path
import json
import csv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SCREENSHOTS = ROOT / "screenshots"
METADATA = ROOT / "models" / "model_metadata.json"
COMPARISON = REPORTS / "model_comparison.csv"

DOCX_PATH = REPORTS / "heart_disease_mlops_report.docx"
PDF_PATH = REPORTS / "heart_disease_mlops_report.pdf"

metadata = json.loads(METADATA.read_text())
with COMPARISON.open(newline="") as f:
    results = list(csv.DictReader(f))

def metric(model, name):
    row = next(r for r in results if r["model"] == model)
    return float(row[name])

def pct(value):
    return f"{value * 100:.2f}%"

def add_page_number(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Heart Disease MLOps | Assignment Report")

def doc_heading(doc, text, level=1):
    doc.add_heading(text, level=level)

def doc_para(doc, text):
    doc.add_paragraph(text)

def doc_image(doc, path, caption, width=5.8):
    path = Path(path)
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width))
        c = doc.add_paragraph(caption)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in c.runs:
            r.italic = True
            r.font.size = Pt(9)

def doc_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
    doc.add_paragraph()

def build_docx():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.65)
    sec.bottom_margin = Inches(.65)
    sec.left_margin = Inches(.7)
    sec.right_margin = Inches(.7)
    add_page_number(sec)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.styles["Title"].font.size = Pt(27)
    doc.styles["Heading 1"].font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("\n\nHEART DISEASE PREDICTION\n").bold = True
    p.add_run("An End-to-End MLOps Implementation").italic = True
    doc.add_paragraph("\n")
    p = doc.add_paragraph("AIMLCZG523 — MLOps Assignment 01")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Reproducible machine learning, experiment tracking,\nAPI serving, containerization, CI/CD and monitoring")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("\n\n")
    p = doc.add_paragraph("Repository: https://github.com/iamkeerthy7/heart-disease-mlops")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph("Prepared: 19 October 2026")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    doc_heading(doc, "Abstract")
    doc_para(doc,
        "This project implements a reproducible machine-learning workflow for binary "
        "classification using the UCI Cleveland Heart Disease dataset. It covers data "
        "inspection and exploratory analysis, preprocessing, model comparison, "
        "cross-validation, MLflow experiment tracking, automated tests, a FastAPI "
        "prediction service, Docker packaging, Kubernetes deployment configuration, "
        "and request monitoring. Logistic Regression was selected using mean "
        "five-fold cross-validation ROC-AUC. The final test-set results are reported "
        "from the project's saved evaluation output. The system is an educational "
        "demonstration and is not validated for clinical decision-making.")
    doc_heading(doc, "1. Objectives and scope")
    doc_para(doc,
        "The goal is to demonstrate the complete lifecycle of a machine-learning "
        "application rather than only fit a classifier. The implementation separates "
        "data processing, training, inference, testing and deployment configuration. "
        "It records experiments, persists a reusable preprocessing-and-model pipeline, "
        "exposes predictions through HTTP, and provides operational metrics.")
    doc_para(doc,
        "The principal deliverables are Python source code, a cleaned dataset, EDA "
        "figures, model artifacts, MLflow records, tests, a CI workflow, a Dockerfile, "
        "Kubernetes manifests, evidence screenshots, and this report.")

    doc_heading(doc, "2. Dataset and target definition")
    doc_para(doc,
        "The project uses the Cleveland heart-disease dataset with 303 observations "
        "and 14 columns in the supplied heart.csv file. Thirteen input features are "
        "used: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, "
        "slope, ca and thal. The original target num is converted to a binary target: "
        "num > 0 indicates the positive class; num = 0 indicates the negative class.")
    doc_para(doc,
        "The project records missing values in ca (4 observations) and thal "
        "(2 observations). Imputation is performed within the preprocessing pipeline "
        "to keep the transformation reproducible and to avoid fitting imputers on "
        "the held-out test data.")
    doc_image(doc, FIGURES / "original_target_balance.png",
              "Figure 1. Distribution of the original target values.")
    doc_image(doc, FIGURES / "binary_class_balance.png",
              "Figure 2. Class balance after binary target conversion.")

    doc_heading(doc, "3. Exploratory data analysis")
    doc_para(doc,
        "EDA artifacts include target distributions, feature histograms, categorical "
        "distributions, a correlation heatmap and model confusion matrices. These "
        "visualizations support inspection of class balance, feature ranges, "
        "relationships between numeric variables and the structure of categorical "
        "features. Correlation is descriptive and does not establish causality.")
    doc_image(doc, FIGURES / "feature_histograms.png",
              "Figure 3. Histograms of numeric feature distributions.")
    doc_image(doc, FIGURES / "categorical_distributions.png",
              "Figure 4. Distributions of categorical features.")
    doc_image(doc, FIGURES / "correlation_heatmap.png",
              "Figure 5. Correlation heatmap for the dataset.")

    doc_heading(doc, "4. Preprocessing and model development")
    doc_para(doc,
        "The preprocessing pipeline uses a ColumnTransformer. Numeric columns are "
        "imputed with the median and scaled. Categorical columns are imputed using "
        "the most frequent value and one-hot encoded. This pipeline is bundled with "
        "the estimator so training and inference apply the same transformations.")
    doc_para(doc,
        "Two classifiers were evaluated: Logistic Regression and Random Forest. "
        "GridSearchCV with stratified five-fold cross-validation was used for "
        "hyperparameter selection. The comparison records accuracy, precision, "
        "recall and ROC-AUC, along with cross-validation means and standard deviations.")
    doc_para(doc,
        "The saved model metadata identifies Logistic Regression as the selected "
        "model because it achieved the higher mean cross-validation ROC-AUC. The "
        "selected configuration records C = 1.0. The Random Forest search selected "
        "200 estimators, maximum depth 5 and minimum leaf size 1.")

    doc_heading(doc, "5. Evaluation results")
    doc_table(doc,
        ["Metric", "Logistic Regression", "Random Forest"],
        [
            ["Test accuracy", pct(metric("logistic_regression", "accuracy")),
             pct(metric("random_forest", "accuracy"))],
            ["Test precision", pct(metric("logistic_regression", "precision")),
             pct(metric("random_forest", "precision"))],
            ["Test recall", pct(metric("logistic_regression", "recall")),
             pct(metric("random_forest", "recall"))],
            ["Test ROC-AUC", f'{metric("logistic_regression", "roc_auc"):.4f}',
             f'{metric("random_forest", "roc_auc"):.4f}'],
            ["Mean CV ROC-AUC", f'{metric("logistic_regression", "cv_roc_auc_mean"):.4f}',
             f'{metric("random_forest", "cv_roc_auc_mean"):.4f}'],
            ["CV ROC-AUC standard deviation",
             f'{metric("logistic_regression", "cv_roc_auc_std"):.4f}',
             f'{metric("random_forest", "cv_roc_auc_std"):.4f}'],
        ])
    doc_para(doc,
        "Both models achieved 88.52% test accuracy, 83.87% precision and 92.86% "
        "recall on this split. Logistic Regression achieved test ROC-AUC 0.9665 "
        "versus 0.9513 for Random Forest. Its mean cross-validation ROC-AUC was "
        "0.9025 versus 0.9020. The cross-validation difference is small, so the "
        "selection should not be interpreted as proof of general superiority.")
    doc_image(doc, FIGURES / "confusion_matrix_logistic_regression.png",
              "Figure 6. Confusion matrix for Logistic Regression.")
    doc_image(doc, FIGURES / "confusion_matrix_random_forest.png",
              "Figure 7. Confusion matrix for Random Forest.")

    doc_heading(doc, "6. Experiment tracking with MLflow")
    doc_para(doc,
        "MLflow is used to track model parameters, evaluation metrics and artifacts "
        "for the training runs. The project stores tracking data locally in the "
        "MLflow database and artifact directory. This makes model comparisons "
        "inspectable and supports repeatable experiments. Screenshots below provide "
        "visual evidence of recorded runs and a run-detail view.")
    doc_image(doc, SCREENSHOTS / "03-mlflow-runs.png",
              "Figure 8. MLflow experiment runs.")
    doc_image(doc, SCREENSHOTS / "03b-mlflow-run-details.png",
              "Figure 9. MLflow run details and logged values.")

    doc_heading(doc, "7. Model persistence and API")
    doc_para(doc,
        "The selected fitted pipeline is persisted as models/best_model.joblib, "
        "with model selection details in models/model_metadata.json. The FastAPI "
        "application provides a health endpoint, a JSON prediction endpoint, "
        "Prometheus metrics and a monitoring dashboard. The prediction response "
        "contains the predicted class, a human-readable label, confidence, class "
        "probabilities and an educational-use disclaimer.")
    doc_para(doc,
        "Input validation rejects malformed or incomplete requests. The reusable "
        "pipeline ensures imputation, scaling and categorical encoding are applied "
        "at inference time using the fitted training transformations.")
    doc_image(doc, SCREENSHOTS / "06-docker-predict.png",
              "Figure 10. Example prediction response from the containerized API.")
    doc_image(doc, SCREENSHOTS / "08-kubernetes-predict.png",
              "Figure 11. Example prediction response through Kubernetes ingress.")

    doc_heading(doc, "8. Automated testing and CI")
    doc_para(doc,
        "The project includes tests for API health, valid prediction requests, "
        "invalid input, missing features, metrics and the dashboard. Preprocessing "
        "tests cover finite transformed values, missing values, unseen categories "
        "and feature partitioning. The recorded verification was 10 passing tests. "
        "Ruff linting and Python compilation checks were also run.")
    doc_para(doc,
        "The GitHub Actions workflow installs dependencies, runs linting and "
        "compilation, executes the test suite, trains the model and uploads "
        "generated artifacts. A failing validation command causes the workflow to "
        "fail, making errors visible rather than silently ignored.")
    doc_image(doc, SCREENSHOTS / "04-github-actions.png",
              "Figure 12. GitHub Actions workflow evidence.")

    doc_heading(doc, "9. Docker and Kubernetes deployment")
    doc_para(doc,
        "The Docker image packages the Python application and model for consistent "
        "execution. The image exposes port 8000. The local container was tested "
        "using host port 8001 because port 8000 was already occupied on the host.")
    doc_image(doc, SCREENSHOTS / "05-docker-build.png",
              "Figure 13. Docker image build evidence.")
    doc_para(doc,
        "Kubernetes manifests define the API Deployment, a LoadBalancer Service "
        "and an NGINX Ingress. The local Minikube deployment reached one ready "
        "replica, and ingress requests to health, prediction, metrics and dashboard "
        "routes were tested. In local Minikube, a LoadBalancer external IP may "
        "remain pending; the ingress address is local and is not a public URL.")
    doc_image(doc, SCREENSHOTS / "07-kubernetes-resources.png",
              "Figure 14. Kubernetes resources and deployment status.")

    doc_heading(doc, "10. Monitoring and request logging")
    doc_para(doc,
        "The API middleware records request method, route, response status and "
        "duration. Prometheus metrics include HTTP request counters, latency "
        "histograms and prediction outcome counters. A dashboard summarizes "
        "request volume, successful predictions, errors, latency and status.")
    doc_image(doc, SCREENSHOTS / "09-prometheus-metrics.png",
              "Figure 15. Prometheus metrics exposed by the API.")
    doc_image(doc, SCREENSHOTS / "10-monitoring-dashboard.png",
              "Figure 16. Application monitoring dashboard.")

    doc_heading(doc, "11. Reproducibility and repository layout")
    doc_para(doc,
        "The repository separates source code, data, model artifacts, reports, "
        "tests, deployment configuration and screenshots. The main setup path is "
        "to create a Python 3.12 environment, install requirements.txt, inspect "
        "the dataset, run the training command, execute pytest and Ruff, and then "
        "start the FastAPI application. Docker and Kubernetes deployment instructions "
        "are documented in the repository README.")
    doc_para(doc,
        "Key paths include data/heart.csv, src/heart_mlops/, tests/, models/, "
        "reports/figures/, deployment/k8s/, .github/workflows/ and screenshots/. "
        "MLflow tracking files and trained artifacts should be preserved when "
        "reproducing the recorded experiment history.")

    doc_heading(doc, "12. Limitations, risks and future work")
    doc_para(doc,
        "This dataset contains only 303 observations and represents a historical "
        "dataset rather than a representative modern clinical population. A single "
        "held-out split and cross-validation cannot establish clinical utility, "
        "calibration, fairness or generalization to other hospitals or populations. "
        "The prediction endpoint is a software demonstration, not medical advice.")
    doc_para(doc,
        "Future improvements include external validation, confidence calibration, "
        "threshold analysis based on explicit use-case costs, drift monitoring, "
        "stronger API authentication and rate limiting, dependency and image "
        "scanning, centralized secrets management, and deployment to a secured "
        "cloud Kubernetes cluster with a publicly accessible endpoint.")
    doc_heading(doc, "Conclusion")
    doc_para(doc,
        "The project demonstrates a complete educational MLOps path from data "
        "inspection through model evaluation, experiment tracking, automated "
        "testing, containerization, local Kubernetes deployment and monitoring. "
        "The selected Logistic Regression pipeline is reproducible and served by "
        "an API. The recorded results are promising for the supplied dataset but "
        "must not be interpreted as evidence of clinical readiness.")
    doc_heading(doc, "Repository and evidence")
    doc_para(doc,
        "Source repository: https://github.com/iamkeerthy7/heart-disease-mlops")
    doc_para(doc,
        "The screenshots referenced in this report are maintained under the "
        "repository's screenshots/ directory. The EDA and confusion-matrix figures "
        "are maintained under reports/figures/. Consult README.md for exact "
        "commands, configuration, and local deployment instructions.")
    doc_heading(doc, "Architecture")
    doc_para(doc,
        "The diagram below summarizes the data, training, experiment tracking, "
        "API serving, CI/CD, container deployment and monitoring workflow.")
    doc_image(doc, FIGURES / "architecture_diagram.png",
              "Figure 17. End-to-end architecture of the implemented MLOps workflow.",
              width=6.0)
    doc.save(DOCX_PATH)

def build_pdf():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"],
                              alignment=TA_CENTER, fontSize=23, leading=28,
                              spaceAfter=18))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"],
                              alignment=TA_CENTER, fontSize=8, leading=10,
                              textColor=colors.grey))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"],
                              textColor=colors.HexColor("#1f4e79"),
                              spaceBefore=8, spaceAfter=7))
    story = []
    def para(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 5))
    def heading(text):
        para(text, "Section")
    def page():
        story.append(PageBreak())
    def image(path, caption, maxw=170*mm, maxh=105*mm):
        path = Path(path)
        if path.exists():
            img = Image(str(path))
            scale = min(maxw/img.imageWidth, maxh/img.imageHeight)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            story.extend([img, Paragraph(caption, styles["Caption"]), Spacer(1, 8)])
    def table(data):
        t = Table(data, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f4e79")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .4, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#eef3f8")]),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.extend([t, Spacer(1, 9)])

    para("HEART DISEASE PREDICTION", "CoverTitle")
    para("<b>An End-to-End MLOps Implementation</b>", "Heading2")
    para("AIMLCZG523 — MLOps Assignment 01")
    para("Reproducible ML, MLflow, automated testing, FastAPI, Docker, Kubernetes and monitoring")
    para("<link href='https://github.com/iamkeerthy7/heart-disease-mlops' color='blue'>Project repository</link>")
    para("Prepared: 19 October 2026")
    page()

    heading("Abstract")
    para("This report documents an educational binary-classification MLOps project "
         "using the UCI Cleveland Heart Disease dataset. The workflow includes EDA, "
         "reusable preprocessing, model comparison, MLflow tracking, automated "
         "tests, FastAPI inference, Docker, Kubernetes manifests and Prometheus "
         "monitoring. Logistic Regression was selected by mean five-fold CV ROC-AUC. "
         "The model is not clinically validated and must not be used for diagnosis.")
    heading("1. Objectives and scope")
    para("The project demonstrates lifecycle engineering around a classifier: "
         "data inspection, training, model persistence, API validation, CI checks, "
         "container packaging, deployment configuration and operational metrics. "
         "The repository contains source, tests, model artifacts, EDA figures, "
         "deployment manifests and evidence screenshots.")
    heading("2. Dataset and target")
    para("The supplied heart.csv contains 303 rows and 14 columns. The 13 predictors "
         "are age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, "
         "slope, ca and thal. The binary target is num > 0. Missing values were "
         "recorded for ca (4) and thal (2). Numeric features use median imputation "
         "and scaling; categorical features use most-frequent imputation and one-hot encoding.")
    image(FIGURES/"original_target_balance.png", "Figure 1. Original target distribution.")
    image(FIGURES/"binary_class_balance.png", "Figure 2. Binary target distribution.")
    page()
    heading("3. Exploratory data analysis")
    para("EDA artifacts include target balance, numeric histograms, categorical "
         "distributions and a correlation heatmap. They help identify class balance, "
         "ranges and associations; correlation does not establish causality.")
    image(FIGURES/"feature_histograms.png", "Figure 3. Numeric feature histograms.")
    image(FIGURES/"categorical_distributions.png", "Figure 4. Categorical distributions.")
    image(FIGURES/"correlation_heatmap.png", "Figure 5. Correlation heatmap.")
    page()
    heading("4. Preprocessing and model development")
    para("A ColumnTransformer keeps preprocessing consistent between training and "
         "inference. Logistic Regression and Random Forest were evaluated with "
         "GridSearchCV and stratified five-fold cross-validation. Metrics include "
         "accuracy, precision, recall and ROC-AUC; cross-validation means and "
         "standard deviations are saved in model_comparison.csv.")
    para("The selected Logistic Regression configuration uses C=1.0. The selected "
         "Random Forest configuration uses 200 trees, maximum depth 5 and minimum "
         "leaf size 1. Selection was based on mean cross-validation ROC-AUC.")
    heading("5. Evaluation")
    table([
        ["Metric", "Logistic Regression", "Random Forest"],
        ["Test accuracy", pct(metric("logistic_regression","accuracy")), pct(metric("random_forest","accuracy"))],
        ["Test precision", pct(metric("logistic_regression","precision")), pct(metric("random_forest","precision"))],
        ["Test recall", pct(metric("logistic_regression","recall")), pct(metric("random_forest","recall"))],
        ["Test ROC-AUC", f'{metric("logistic_regression","roc_auc"):.4f}', f'{metric("random_forest","roc_auc"):.4f}'],
        ["Mean CV ROC-AUC", f'{metric("logistic_regression","cv_roc_auc_mean"):.4f}', f'{metric("random_forest","cv_roc_auc_mean"):.4f}'],
        ["CV ROC-AUC std", f'{metric("logistic_regression","cv_roc_auc_std"):.4f}', f'{metric("random_forest","cv_roc_auc_std"):.4f}'],
    ])
    para("Both classifiers achieved 88.52% accuracy, 83.87% precision and 92.86% "
         "recall on the held-out test set. Logistic Regression achieved ROC-AUC "
         "0.9665, compared with 0.9513 for Random Forest. Mean CV ROC-AUC was "
         "0.9025 versus 0.9020, a small difference that does not establish "
         "general superiority.")
    image(FIGURES/"confusion_matrix_logistic_regression.png", "Figure 6. Logistic Regression confusion matrix.")
    image(FIGURES/"confusion_matrix_random_forest.png", "Figure 7. Random Forest confusion matrix.")
    page()
    heading("6. MLflow experiment tracking")
    para("MLflow records run parameters, metrics and artifacts, supporting inspection "
         "of experiments. Tracking data is stored locally in the project database "
         "and artifact directory.")
    image(SCREENSHOTS/"03-mlflow-runs.png", "Figure 8. MLflow experiment runs.")
    image(SCREENSHOTS/"03b-mlflow-run-details.png", "Figure 9. MLflow run details.")
    heading("7. Model persistence and API")
    para("The selected pipeline is stored as models/best_model.joblib; "
         "models/model_metadata.json records the selected model, metric, target "
         "definition and feature list. FastAPI exposes health, prediction, metrics "
         "and dashboard endpoints. Predictions include class, label, confidence, "
         "probabilities and an educational-use disclaimer.")
    image(SCREENSHOTS/"06-docker-predict.png", "Figure 10. Container API prediction.")
    image(SCREENSHOTS/"08-kubernetes-predict.png", "Figure 11. Prediction through Kubernetes ingress.")
    page()
    heading("8. Testing and CI")
    para("The project has tests for API health, valid predictions, invalid age, "
         "missing features, metrics and dashboard behavior, plus preprocessing "
         "checks for finite values, missing data, unseen categories and feature "
         "partitioning. Recorded verification was 10 passing tests; Ruff and "
         "compile checks also passed. GitHub Actions runs validation, tests and "
         "training, then uploads artifacts.")
    image(SCREENSHOTS/"04-github-actions.png", "Figure 12. GitHub Actions evidence.")
    heading("9. Docker and Kubernetes")
    para("The Docker image packages the API and model and exposes port 8000. "
         "The container was tested on host port 8001 because port 8000 was occupied. "
         "Kubernetes manifests configure a Deployment, LoadBalancer Service and "
         "NGINX Ingress. Local Minikube showed a ready replica and successful "
         "tests of health, prediction, metrics and dashboard routes. The local "
         "ingress address is not a public endpoint.")
    image(SCREENSHOTS/"05-docker-build.png", "Figure 13. Docker build evidence.")
    image(SCREENSHOTS/"07-kubernetes-resources.png", "Figure 14. Kubernetes resources.")
    page()
    heading("10. Monitoring and logging")
    para("Request middleware logs method, path, status and duration. Prometheus "
         "counters and latency histograms capture HTTP activity and prediction "
         "outcomes. The dashboard summarizes request volume, successful predictions, "
         "errors, latency and service status.")
    image(SCREENSHOTS/"09-prometheus-metrics.png", "Figure 15. Prometheus metrics.")
    image(SCREENSHOTS/"10-monitoring-dashboard.png", "Figure 16. Monitoring dashboard.")
    heading("11. Reproducibility")
    para("Key repository paths are data/heart.csv, src/heart_mlops/, tests/, "
         "models/, reports/figures/, deployment/k8s/, .github/workflows/ and "
         "screenshots/. Follow README.md for environment setup, training, tests, "
         "MLflow, API, Docker and Kubernetes commands. Preserve local MLflow "
         "records and model artifacts when reproducing experiment history.")
    heading("12. Limitations and future work")
    para("The dataset is small (303 observations) and historical. A held-out split "
         "and cross-validation do not establish clinical validity, calibration, "
         "fairness or generalization to other populations. Future work includes "
         "external validation, calibration, threshold analysis, drift monitoring, "
         "authentication, rate limiting, image/dependency scanning, secrets "
         "management and secure cloud deployment.")
    page()
    heading("Architecture")
    para("The diagram summarizes the data, training, experiment tracking, API serving, "
         "CI/CD, container deployment and monitoring workflow.")
    image(FIGURES/"architecture_diagram.png",
          "Figure 17. End-to-end architecture of the implemented MLOps workflow.",
          maxw=170*mm, maxh=95*mm)

    heading("Conclusion")
    para("The implementation demonstrates the main MLOps stages from data analysis "
         "to monitored API serving. Logistic Regression was selected based on "
         "mean cross-validation ROC-AUC. Results apply to this dataset and split "
         "only; this system is an educational demonstration, not medical advice.")
    heading("Repository and evidence")
    para("<link href='https://github.com/iamkeerthy7/heart-disease-mlops' color='blue'>"
         "https://github.com/iamkeerthy7/heart-disease-mlops</link>")
    para("Evidence screenshots are under screenshots/; EDA and confusion matrix "
         "figures are under reports/figures/. Consult README.md for commands and setup.")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0]/2, 10*mm,
                                 f"Heart Disease MLOps | Page {doc.page}")
        canvas.restoreState()

    pdf = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                            rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=15*mm, bottomMargin=17*mm,
                            title="Heart Disease MLOps Assignment Report",
                            author="MLOps Assignment")
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)

if __name__ == "__main__":
    build_docx()
    build_pdf()
    print(f"Created: {DOCX_PATH}")
    print(f"Created: {PDF_PATH}")
