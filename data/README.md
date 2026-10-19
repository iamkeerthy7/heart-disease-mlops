# Heart Disease Dataset

Source: UCI Machine Learning Repository
Dataset: Heart Disease — Cleveland processed dataset

Source URL:
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

The raw dataset contains 303 records and 14 columns:
13 input features and one target column.

Missing values in the original data are represented by '?'.
The columns `ca` and `thal` contain missing values.

For binary classification:
- Target 0: no heart disease recorded
- Targets 1, 2, 3, and 4: heart disease recorded

The raw target is named `num`. The derived binary target will be
named `target` in the model pipeline.

This dataset is intended for educational machine learning work,
not for clinical diagnosis.
