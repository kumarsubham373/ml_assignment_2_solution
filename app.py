"""Interactive evaluator for the five saved academic-outcome classifiers.

Upload a CSV that already contains the 36 input features and the Target
column. Each model is loaded from model/*.joblib and is never refit here.
"""

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluation import matrix, report_frame, score_predictions
from schema import CLASS_ORDER, FEATURE_COLUMNS, MODEL_FILES, TARGET_COLUMN

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST = ROOT / "test_data.csv"

st.set_page_config(
    page_title="Academic Outcome Bench",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, #e7f2ef 0%, transparent 55%),
            radial-gradient(900px 400px at 100% 0%, #f8efe4 0%, transparent 50%),
            #f7f5f1;
    }
    .hero {
        background: #16352f;
        color: #f6f3ec;
        border-radius: 18px;
        padding: 28px 32px 22px 32px;
        margin-bottom: 16px;
    }
    .hero h1 {
        color: #f6f3ec;
        font-size: 2rem;
        margin: 0 0 8px 0;
        letter-spacing: -0.03em;
    }
    .hero p { margin: 0; color: #d5e4df; font-size: 1.02rem; }
    .note {
        background: #fffdf8;
        border: 1px solid #e4dccf;
        border-radius: 12px;
        padding: 12px 14px;
        color: #3d3832;
    }
    div[data-testid="stMetric"] {
        background: #fffdf8;
        border: 1px solid #e4dccf;
        border-radius: 12px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run python train.py before starting the app."
            )
        loaded[name] = joblib.load(path)
    return loaded


def read_table(uploaded_file):
    if uploaded_file is None:
        frame = pd.read_csv(BUNDLED_TEST)
        source = "Bundled held-out test set (test_data.csv)"
    else:
        raw = uploaded_file.getvalue()
        separator = ";" if raw.count(b";") > raw.count(b",") else ","
        frame = pd.read_csv(io.BytesIO(raw), sep=separator)
        source = uploaded_file.name
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    return frame, source


def validate(frame):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return f"The file is missing: {', '.join(missing)}"
    if frame[required].isna().any().any():
        return "The required columns contain missing values. Fill them before scoring."
    unknown = sorted(set(frame[TARGET_COLUMN].astype(str)) - set(CLASS_ORDER))
    if unknown:
        return (
            "Target must be Dropout, Enrolled, or Graduate. "
            f"Found extra labels: {', '.join(unknown)}"
        )
    if len(frame) < 2:
        return "Upload at least two labelled rows so the metrics are defined."
    return None


def score_model(pipeline, frame):
    features = frame[FEATURE_COLUMNS]
    actual = frame[TARGET_COLUMN].astype(str)
    predicted = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    scores = score_predictions(
        actual,
        predicted,
        probabilities,
        pipeline.named_steps["classifier"].classes_,
    )
    return scores, actual, predicted


def draw_confusion(actual, predicted, title):
    counts = matrix(actual, predicted)
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="YlGn",
        cbar=False,
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
        ax=axis,
        linewidths=0.6,
        linecolor="#f7f5f1",
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>Academic Outcome Bench</h1>
            <h3>By: Shubham Kumar</h3>
            <h3>ID: 2026AJ05228</h3>
            <p>Five saved classifiers score a labelled student file.
            Nothing is retrained in the browser.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader(
            "Upload a labelled CSV",
            type=["csv"],
            help="Keep the 36 input features and the Target column. "
            "Leave this empty to score the bundled test set.",
        )
        st.header("Model")
        selected = st.selectbox("Classifier", list(MODEL_FILES))
        st.markdown(
            """
            **How a score is produced**

            The file is checked for the 36 original columns and `Target`.
            Logistic regression, the decision tree, k-nearest neighbours, and the
            support vector machine one-hot encode nominal codes and scale the
            other columns. Gaussian naive Bayes scales the original 36 columns.
            Metrics are computed on this file only. The saved pipeline is not refit.
            """
        )

    frame, source = read_table(uploaded)
    problem = validate(frame)
    st.caption(f"Scoring **{source}** · {len(frame):,} rows · model file `{MODEL_FILES[selected]}`")
    if problem:
        st.error(problem)
        st.dataframe(frame.head(8), width="stretch")
        return

    scores, actual, predicted = score_model(models[selected], frame)

    metric_columns = st.columns(6)
    labels = [
        ("Accuracy", "accuracy"),
        ("AUC", "auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("MCC", "mcc"),
    ]
    for column, (label, key) in zip(metric_columns, labels):
        column.metric(label, f"{scores[key]:.4f}")

    st.markdown(
        '<div class="note">Precision, recall, and F1 are macro averages over '
        "Dropout, Enrolled, and Graduate. AUC is weighted one-vs-rest ROC AUC. "
        "The pipeline loaded from disk is used as-is.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Confusion matrix")
        figure = draw_confusion(actual, predicted, selected)
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Classification report")
        report = pd.DataFrame(report_frame(actual, predicted))
        report[["precision", "recall", "f1"]] = report[["precision", "recall", "f1"]].round(4)
        report["support"] = report["support"].round(0).astype(int)
        st.dataframe(report, width="stretch", hide_index=True)
        counts = actual.value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
        st.write(
            "Rows in this file — "
            + ", ".join(f"{label}: {int(counts[label])}" for label in CLASS_ORDER)
        )

    st.subheader("All five models on this file")
    rows = []
    for name, pipeline in models.items():
        model_scores, _, _ = score_model(pipeline, frame)
        rows.append({"ML Model Name": name, **model_scores})
    comparison = pd.DataFrame(rows)
    comparison = comparison.rename(
        columns={
            "accuracy": "Accuracy",
            "auc": "AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "mcc": "MCC",
        }
    )
    for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        comparison[column] = comparison[column].round(4)
    best_f1 = comparison["F1"].max()
    highlight = comparison.style.highlight_max(subset=["F1"], color="#d9efe4")
    st.dataframe(highlight, width="stretch", hide_index=True)
    winner = comparison.loc[comparison["F1"] == best_f1, "ML Model Name"].iloc[0]
    st.success(f"Highest macro F1 on this file: {winner}.")

    with st.expander("Preview of the scored rows"):
        preview = frame[FEATURE_COLUMNS[:6] + [TARGET_COLUMN]].head(12).copy()
        preview.insert(0, "Predicted", predicted[: len(preview)])
        st.dataframe(preview, width="stretch")


if __name__ == "__main__":
    main()
