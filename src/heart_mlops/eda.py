from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_FILE = Path("data/heart.csv")
FIGURE_DIR = Path("reports/figures")
REPORT_DIR = Path("reports")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
]


def load_data() -> pd.DataFrame:
    """Load the raw Cleveland dataset and identify missing values."""
    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    df = load_data()

    print("\n--- Dataset shape ---")
    print(df.shape)

    print("\n--- First five rows ---")
    print(df.head())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Descriptive statistics ---")
    print(df.describe().round(2))

    print("\n--- Original target distribution ---")
    print(df["num"].value_counts(dropna=False).sort_index())

    # Save missing-value summary
    df.isnull().sum().rename("missing_count").to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    # 1. Feature distributions
    df.hist(figsize=(16, 12), bins=20, edgecolor="black")
    plt.suptitle("Heart Disease Dataset: Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_histograms.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 3. Original target class balance
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="num")
    plt.title("Original Heart Disease Target Distribution")
    plt.xlabel("Original target (0 = absent, 1–4 = present)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "original_target_balance.png", dpi=150)
    plt.close()

    # 4. Categorical feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for ax, column in zip(axes.flat, CATEGORICAL_COLUMNS):
        sns.countplot(data=df, x=column, ax=ax)
        ax.set_title(f"Distribution: {column}")
        ax.tick_params(axis="x", rotation=30)

    for ax in axes.flat[len(CATEGORICAL_COLUMNS):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "categorical_distributions.png", dpi=150)
    plt.close()

    # Convert target to binary for the classification problem.
    binary_df = df.dropna(subset=["num"]).copy()
    binary_df["target"] = (binary_df["num"] > 0).astype(int)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=binary_df, x="target")
    plt.title("Binary Heart Disease Class Balance")
    plt.xlabel("Target (0 = absence, 1 = presence)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "binary_class_balance.png", dpi=150)
    plt.close()

    print("\n--- Binary target distribution ---")
    print(binary_df["target"].value_counts().sort_index())

    print(f"\nEDA figures saved to: {FIGURE_DIR.resolve()}")
    print(f"EDA summaries saved to: {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
