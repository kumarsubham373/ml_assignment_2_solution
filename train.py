from pathlib import Path
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import score_predictions
from schema import (
    BINARY_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_FILES,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data.csv"


def load_raw_dataset(path):
    frame = pd.read_csv(path, sep=";")
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; this script does not impute.")
    return frame


def make_preprocessor():
    """One-hot nominal codes, then standardise grades, rates, and binary flags."""
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS + BINARY_COLUMNS),
        ],
        remainder="drop",
    )


def make_gaussian_preprocessor():
    """Standardise the original columns for GaussianNB.

    One-hot indicators are Bernoulli, not Gaussian. Feeding them to GaussianNB
    collapses the model, so this classifier keeps the 36 raw fields.
    """
    return ColumnTransformer(
        transformers=[("all", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )


def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="minkowski",
            p=2,
        ),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def main():
    warnings.filterwarnings(
        "ignore",
        message="The `probability` parameter was deprecated",
        category=FutureWarning,
    )
    frame = load_raw_dataset(DATA_PATH)
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    test_frame = x_test.copy()
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_frame.to_csv(TEST_PATH, index=False)

    print(f"Rows: {len(frame)} | Features: {len(FEATURE_COLUMNS)}")
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print("Class counts (full data):")
    print(target.value_counts().to_string())
    print()
    header = f"{'Model':<24}{'Accuracy':>10}{'AUC':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'MCC':>10}"
    print(header)
    print("-" * len(header))

    metrics = {}
    for name, estimator in make_models().items():
        preprocessor = (
            make_gaussian_preprocessor()
            if name == "Naive Bayes"
            else make_preprocessor()
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)
        scores = score_predictions(
            y_test,
            predicted,
            probabilities,
            pipeline.named_steps["classifier"].classes_,
        )
        metrics[name] = scores
        joblib.dump(pipeline, MODEL_DIR / MODEL_FILES[name])
        print(
            f"{name:<24}"
            f"{scores['accuracy']:10.4f}"
            f"{scores['auc']:10.4f}"
            f"{scores['precision']:12.4f}"
            f"{scores['recall']:10.4f}"
            f"{scores['f1']:10.4f}"
            f"{scores['mcc']:10.4f}"
        )

    joblib.dump(
        {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "metric_notes": {
                "precision_recall_f1": "macro average over Dropout, Enrolled, Graduate",
                "auc": "weighted one-vs-rest ROC AUC from predict_proba",
                "mcc": "multiclass Matthews correlation coefficient",
            },
            "models": metrics,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "class_counts": target.value_counts().to_dict(),
        },
        MODEL_DIR / "training_record.joblib",
    )
    print()
    print(f"Saved models in {MODEL_DIR}")
    print(f"Saved held-out test rows in {TEST_PATH}")


if __name__ == "__main__":
    main()
