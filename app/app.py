import streamlit as st
import pandas as pd
import os
from pathlib import Path
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "processed" / "segmented_buyers.csv"
MODEL_PATH = BASE_DIR / "models" / "buyer_segmentation_model.pkl"
ANALYSIS_PATH = BASE_DIR / "data" / "processed" / "segment_analysis.csv"

# =========================================================
# DEBUGGING (REMOVE LATER)
# =========================================================

with st.sidebar:
    st.write("Current Directory:")
    st.code(os.getcwd())

# =========================================================
# DATASET CHECK
# =========================================================

if not DATA_PATH.exists():
    st.error("❌ Buyer dataset was not found.")

    st.write("Expected location:")
    st.code(str(DATA_PATH))

    st.write("Available CSV files:")

    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".csv"):
                st.write(os.path.join(root, file))

    st.stop()

# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error("Unable to load dataset.")
    st.exception(e)
    st.stop()

# =========================================================
# FIND SEGMENT COLUMN
# =========================================================

possible_segment_columns = [
    "buyer_segment",
    "segment",
    "Segment",
    "cluster",
    "Cluster"
]

segment_col = None

for col in possible_segment_columns:
    if col in df.columns:
        segment_col = col
        break

# =========================================================
# DASHBOARD
# =========================================================

st.title("🏠 Real Estate Buyer Intelligence")

st.success("Dataset Loaded Successfully")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Buyers", len(df))

with col2:
    if segment_col:
        st.metric(
            "Segments",
            df[segment_col].nunique()
        )

with col3:
    st.metric(
        "Columns",
        df.shape[1]
    )

# =========================================================
# SEGMENT CHART
# =========================================================

if segment_col:

    segment_counts = (
        df[segment_col]
        .value_counts()
        .sort_index()
    )

    chart_df = pd.DataFrame({
        "Segment": segment_counts.index.astype(str),
        "Buyers": segment_counts.values
    })

    fig = px.bar(
        chart_df,
        x="Segment",
        y="Buyers",
        text="Buyers",
        title="Buyer Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# DATASET VIEW
# =========================================================

st.subheader("Dataset")

st.dataframe(
    df,
    use_container_width=True
)