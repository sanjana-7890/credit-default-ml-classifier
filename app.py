from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "credit_default.csv"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
TARGET = "default_payment_next_month"
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "support_vector_machine": "Support Vector Machine",
}

st.set_page_config(page_title="Credit Default Model Lab", layout="wide")
st.title("Credit Default Model Lab")
st.caption("Credit risk classification | UCI Default of Credit Card Clients")
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #10171a !important;
        color: #e7eeec !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
    [data-testid="stSidebar"] { background-color: #151e22 !important; }
    div.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #eef4f2 !important; font-weight: 650; }
    p, label, [data-testid="stCaptionContainer"] { color: #c4cfcd !important; }
    [data-testid="stMetric"] { background: transparent !important; border: 0; padding: 0.45rem 0.15rem; }
    [data-testid="stMetricLabel"] p { color: #aab8b5 !important; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: #eef4f2 !important; font-size: 1.7rem; }
    .preview-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .preview-table th { color: #b6c4c0; text-align: left; padding: 0.65rem; border-bottom: 1px solid #344348; }
    .preview-table td { color: #e7eeec; padding: 0.65rem; border-bottom: 1px solid #273438; }
    .preview-table tr:nth-child(even) { background-color: #151e22; }
    [data-testid="stFileUploader"] section { background: #182226 !important; border: 1px dashed #405257; border-radius: 6px; }
    [data-baseweb="select"] > div { background-color: #182226 !important; color: #e7eeec !important; border-color: #405257 !important; }
    [data-testid="stAlert"], [data-testid="stAlertContainer"] {
        background-color: #1b292d !important;
        color: #e7eeec !important;
        border-color: #405257 !important;
    }
    [data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] * {
        color: #dce7e4 !important;
    }
    [data-testid="stDownloadButton"] button, .stButton button { border-radius: 6px; background-color: #203136 !important; color: #e7eeec !important; border-color: #405257 !important; }
    [data-testid="stFileUploader"] button {
        background-color: #203136 !important;
        color: #e7eeec !important;
        border-color: #405257 !important;
    }
    [data-testid="stFileUploader"] button p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: #c4cfcd !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_overview_data(data_path: str, modified_time_ns: int) -> pd.DataFrame:
    return pd.read_csv(data_path)


@st.cache_resource
def load_model(model_path: str, modified_time_ns: int):
    return joblib.load(model_path)


def build_pdf_report(model_label, row_count, metrics, matrix, target, predictions):
    buffer = BytesIO()
    report_text = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["No default", "Default"],
        zero_division=0,
    )

    with PdfPages(buffer) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        layout = figure.add_gridspec(
            4,
            1,
            height_ratios=[0.8, 1.4, 4.0, 2.4],
            hspace=0.45,
        )

        heading_axis = figure.add_subplot(layout[0])
        heading_axis.axis("off")
        heading_axis.text(
            0.02,
            0.85,
            f"{model_label} Evaluation Report",
            fontsize=18,
            fontweight="bold",
            va="top",
        )
        heading_axis.text(
            0.02,
            0.2,
            f"Dataset: UCI Default of Credit Card Clients | Labeled test rows: {row_count}",
            fontsize=10,
            va="top",
        )

        metrics_axis = figure.add_subplot(layout[1])
        metrics_axis.axis("off")
        metrics_axis.set_title("Evaluation metrics", loc="left", fontsize=12, pad=8)
        metric_table = metrics_axis.table(
            cellText=[[name, f"{score:.4f}"] for name, score in metrics],
            colLabels=["Metric", "Score"],
            cellLoc="left",
            loc="center",
            colWidths=[0.45, 0.2],
        )
        metric_table.auto_set_font_size(False)
        metric_table.set_fontsize(10)
        metric_table.scale(1, 1.25)

        matrix_axis = figure.add_subplot(layout[2])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No default", "Default"],
            yticklabels=["No default", "Default"],
            ax=matrix_axis,
        )
        matrix_axis.set_title("Confusion matrix", loc="left", fontsize=12, pad=10)
        matrix_axis.set_xlabel("Predicted label")
        matrix_axis.set_ylabel("Actual label")

        report_axis = figure.add_subplot(layout[3])
        report_axis.axis("off")
        report_axis.set_title("Classification report", loc="left", fontsize=12, pad=8)
        report_axis.text(0.02, 0.92, report_text, family="monospace", fontsize=9, va="top")

        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    return buffer.getvalue()


if not RAW_DATA_PATH.is_file():
    st.error(f"Dataset preview file not found: {RAW_DATA_PATH}")
    st.stop()

try:
    overview_data = load_overview_data(
        str(RAW_DATA_PATH), RAW_DATA_PATH.stat().st_mtime_ns
    )
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the dataset: {error}")
    st.stop()

if TARGET not in overview_data.columns:
    st.error(f"The dataset is missing the target column {TARGET!r}.")
    st.stop()

feature_names = [
    column for column in overview_data.columns if column not in {"ID", TARGET}
]
class_counts = (
    overview_data[TARGET]
    .astype(int)
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(index={0: "No default", 1: "Default"})
)
default_rate = overview_data[TARGET].astype(int).mean()

st.header("Dataset overview")
overview_metrics = st.columns(3)
overview_metrics[0].metric("Records", f"{len(overview_data):,}")
overview_metrics[1].metric("Input features", f"{len(feature_names)}")
overview_metrics[2].metric("Default rate", f"{default_rate:.1%}")

preview_column, balance_column = st.columns([1.6, 1])
with preview_column:
    st.subheader("Sample records")
    preview_columns = feature_names[:4] + [TARGET]
    preview = overview_data.loc[:, preview_columns].head(10).rename(
        columns={TARGET: "DEFAULT"}
    )
    st.markdown(
        preview.to_html(index=False, border=0, classes="preview-table", escape=True),
        unsafe_allow_html=True,
    )
with balance_column:
    st.subheader("Target distribution")
    chart_counts = class_counts.loc[["Default", "No default"]]
    figure, axis = plt.subplots(figsize=(4.4, 2.5), facecolor="#10171a")
    axis.set_facecolor("#10171a")
    bars = axis.barh(chart_counts.index, chart_counts.values, color="#55D6BE", height=0.56)
    axis.set_xlim(0, chart_counts.max() * 1.16)
    axis.set_xlabel("Clients", color="#c4cfcd")
    axis.tick_params(colors="#c4cfcd", axis="y", length=0)
    axis.tick_params(colors="#c4cfcd", axis="x")
    axis.xaxis.grid(color="#2a3a3f", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, count in zip(bars, chart_counts.values):
        axis.text(
            count + chart_counts.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            color="#e7eeec",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with st.expander("Dataset details"):
    st.write(
        "Public binary classification dataset from the UCI Machine Learning Repository. "
        "The target indicates whether a client defaulted on a payment next month."
    )
    st.write(
        "Target labels: `0` = no default, `1` = default. "
        "The ID column is not used as a model feature."
    )
    st.write("Input features:")
    st.write(", ".join(feature_names))

if TEST_DATA_PATH.is_file():
    st.download_button(
        "Download labeled test data",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
else:
    st.warning("The project holdout file is missing. Run train_models.py to create it.")

st.divider()
st.header("Evaluate a model")

uploaded_file = st.file_uploader(
    "Step 1: Upload labeled test data (CSV)",
    type=["csv"],
)
if uploaded_file is None:
    st.info("Upload a labeled test CSV to begin the evaluation.")
    st.stop()

try:
    evaluation_data = pd.read_csv(uploaded_file)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the test CSV: {error}")
    st.stop()

if TARGET not in evaluation_data.columns:
    st.error(f"The CSV must include the target column {TARGET!r}.")
    st.stop()

if set(evaluation_data[TARGET].astype(int).unique()) != {0, 1}:
    st.error("The target column must contain both binary labels 0 and 1.")
    st.stop()

st.success(f"Test data uploaded: {len(evaluation_data):,} labeled rows.")
st.info("The test data is ready. Now select the classifier you want to evaluate.")

model_paths = sorted(MODEL_DIR.glob("*.joblib"))
if not model_paths:
    st.error("No saved models found. Run train_models.py first.")
    st.stop()

selected_path = st.selectbox(
    "Step 2: Select a machine-learning algorithm",
    model_paths,
    format_func=lambda path: MODEL_LABELS.get(
        path.stem, path.stem.replace("_", " ").title()
    ),
    index=None,
    placeholder="Choose a classifier",
)
if selected_path is None:
    st.stop()

model = load_model(str(selected_path), selected_path.stat().st_mtime_ns)
model_features = list(getattr(model, "feature_names_in_", []))
if not model_features:
    st.error("The selected model does not include its expected feature names.")
    st.stop()

missing_features = [name for name in model_features if name not in evaluation_data.columns]
if missing_features:
    st.error(f"The CSV is missing required features: {', '.join(missing_features)}")
    st.stop()

features = evaluation_data.loc[:, model_features]
target = evaluation_data[TARGET].astype(int)
if set(target.unique()) != {0, 1}:
    st.error("The target column must contain both binary labels 0 and 1.")
    st.stop()

predictions = model.predict(features)
class_labels = list(model.classes_)
if hasattr(model, "predict_proba"):
    scores = model.predict_proba(features)[:, class_labels.index(1)]
elif hasattr(model, "decision_function"):
    scores = model.decision_function(features)
else:
    st.error("The selected model cannot provide scores for AUC calculation.")
    st.stop()

metric_values = {
    "Accuracy": accuracy_score(target, predictions),
    "AUC": roc_auc_score(target, scores),
    "Precision": precision_score(target, predictions, zero_division=0),
    "Recall": recall_score(target, predictions, zero_division=0),
    "F1": f1_score(target, predictions, zero_division=0),
    "MCC": matthews_corrcoef(target, predictions),
}
matrix = confusion_matrix(target, predictions, labels=[0, 1])
selected_model_label = MODEL_LABELS.get(
    selected_path.stem, selected_path.stem.replace("_", " ").title()
)

st.subheader(selected_model_label)
st.caption(f"Results for {len(evaluation_data):,} labeled test rows.")
metric_columns = st.columns(3)
for index, (metric_name, metric_score) in enumerate(metric_values.items()):
    with metric_columns[index % 3]:
        st.metric(metric_name, f"{metric_score:.4f}")

st.subheader("Confusion matrix")
plot_columns = st.columns([1, 1.2, 1])
with plot_columns[1]:
    figure, axis = plt.subplots(figsize=(4.2, 3.1), facecolor="#0B1014")
    axis.set_facecolor("#0B1014")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        cbar=False,
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
        ax=axis,
    )
    axis.set_xlabel("Predicted label", color="#E8F0F2")
    axis.set_ylabel("Actual label", color="#E8F0F2")
    axis.tick_params(colors="#E8F0F2")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

pdf_report = build_pdf_report(
    selected_model_label,
    len(evaluation_data),
    metric_values.items(),
    matrix,
    target,
    predictions,
)
st.download_button(
    f"Download {selected_model_label} report (PDF)",
    data=pdf_report,
    file_name=f"{selected_path.stem}_evaluation_report.pdf",
    mime="application/pdf",
)
