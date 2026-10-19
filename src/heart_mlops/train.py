import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = Path("data/heart.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def load_dataset():
    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["num"]).copy()
    y = (df.pop("num") > 0).astype(int)
    X = df

    return X, y


def build_preprocessor(scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "logistic_regression": (
            Pipeline([
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]),
            {
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "random_forest": (
            Pipeline([
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                )),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    mlflow.set_experiment("heart-disease-classification")

    results = []
    best_score = -1.0
    best_name = None
    best_model = None

    for name, (pipeline, param_grid) in candidates.items():
        print(f"\nTraining and tuning: {name}")

        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)
        model = search.best_estimator_

        cv_metrics = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
            n_jobs=-1,
        )

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        for metric in ["accuracy", "precision", "recall", "roc_auc"]:
            metrics[f"cv_{metric}_mean"] = float(
                cv_metrics[f"test_{metric}"].mean()
            )
            metrics[f"cv_{metric}_std"] = float(
                cv_metrics[f"test_{metric}"].std()
            )

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_param("best_params", json.dumps(search.best_params_))
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_metrics(metrics)

            results.append({
                "model": name,
                "best_params": json.dumps(search.best_params_),
                **metrics,
            })

            print("Best parameters:", search.best_params_)
            print("Test metrics:", metrics)
            print("\nClassification report:")
            print(classification_report(y_test, predictions, zero_division=0))

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                confusion_matrix(y_test, predictions),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"Confusion Matrix: {name}")
            fig.tight_layout()

            plot_path = FIGURE_DIR / f"confusion_matrix_{name}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(plot_path))
            mlflow.sklearn.log_model(
                model,
                name="heart-disease-model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    import joblib
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump({
            "model_name": best_name,
            "selection_metric": "mean cross-validation ROC-AUC",
            "best_cv_roc_auc": float(best_score),
            "target_definition": "num > 0",
            "features": list(X.columns),
        }, f, indent=2)

    print("\n--- Model comparison ---")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
