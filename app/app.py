import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background-color: #f5f7fb;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }


    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }


    /* Dashboard cards */

    .dashboard-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }


    .card-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
    }


    .card-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }


    .segment-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }


    .segment-number {
        font-size: 28px;
        font-weight: 800;
    }


    .segment-label {
        color: #6b7280;
        font-size: 14px;
    }


    /* Section headings */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 10px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #6b7280;
        padding: 30px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = "data/processed/segmented_buyers.csv"

MODEL_PATH = "models/buyer_segmentation_model.pkl"

ANALYSIS_PATH = "data/processed/segment_analysis.csv"


# =========================================================
# SEGMENT NAMES
# =========================================================

segment_names = {
    1: "K-Means Segment 1",
    2: "K-Means Segment 2",
    3: "K-Means Segment 3"
}


def get_segment_name(segment):

    try:

        number = int(segment)

        return segment_names.get(
            number,
            f"K-Means Segment {number}"
        )

    except (ValueError, TypeError):

        return str(segment)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;">

    <div style="font-size:55px;">🏠</div>

    <h2>Real Estate</h2>

    <p>Buyer Intelligence System</p>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "📌 Navigation",
    [
        "🏠 Dashboard",
        "🎯 Buyer Segments",
        "🔮 Prediction",
        "📊 Dataset"
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    ### 🤖 Machine Learning

    **Algorithm:** K-Means

    **Clusters:** 3

    **Scaling:** StandardScaler

    **Random State:** 42
    """
)


st.sidebar.markdown("---")


st.sidebar.info(
    "This application analyzes real estate "
    "buyers using machine learning and "
    "K-Means clustering."
)


# =========================================================
# CHECK DATA
# =========================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "❌ Buyer dataset was not found."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        DATA_PATH
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(
        "Unable to load the dataset."
    )

    st.write(e)

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


for column in possible_segment_columns:

    if column in df.columns:

        segment_col = column

        break


# =========================================================
# NUMERIC COLUMNS
# =========================================================

numeric_columns = (
    df
    .select_dtypes(include="number")
    .columns
    .tolist()
)


if segment_col in numeric_columns:

    numeric_columns.remove(segment_col)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">🏠 Real Estate Buyer Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered buyer segmentation dashboard using K-Means clustering'
        '</div>',
        unsafe_allow_html=True
    )


    st.success(
        "✅ Buyer segmentation dataset loaded successfully!"
    )


    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📌 Business Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="dashboard-card">

            <div class="card-title">
            👥 TOTAL BUYERS
            </div>

            <div class="card-value">
            {len(df):,}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        segments_count = (
            df[segment_col].nunique()
            if segment_col
            else 0
        )

        st.markdown(
            f"""
            <div class="dashboard-card">

            <div class="card-title">
            🎯 SEGMENTS
            </div>

            <div class="card-value">
            {segments_count}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="dashboard-card">

            <div class="card-title">
            📊 FEATURES
            </div>

            <div class="card-value">
            {len(numeric_columns)}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="dashboard-card">

            <div class="card-title">
            🤖 MODEL
            </div>

            <div class="card-value">
            K-Means
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # SEGMENT DATA
    # -----------------------------------------------------

    if segment_col:

        segment_counts = (
            df[segment_col]
            .value_counts()
            .sort_index()
        )


        chart_df = pd.DataFrame({
            "Segment": [
                get_segment_name(x)
                for x in segment_counts.index
            ],
            "Buyers": segment_counts.values
        })


        # -------------------------------------------------
        # BAR + PIE
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🎯 Buyer Segmentation Overview</div>',
            unsafe_allow_html=True
        )


        chart_col1, chart_col2 = st.columns(2)


        with chart_col1:

            fig_bar = px.bar(
                chart_df,
                x="Segment",
                y="Buyers",
                text="Buyers",
                title="Number of Buyers by Segment"
            )

            fig_bar.update_traces(
                textposition="outside"
            )

            fig_bar.update_layout(
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )


        with chart_col2:

            fig_pie = px.pie(
                chart_df,
                names="Segment",
                values="Buyers",
                hole=0.45,
                title="Buyer Segment Distribution"
            )

            fig_pie.update_layout(
                height=420,
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )


        # -------------------------------------------------
        # SEGMENT CARDS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📍 Segment Summary</div>',
            unsafe_allow_html=True
        )


        segment_values = list(
            segment_counts.items()
        )


        segment_cols = st.columns(
            max(1, len(segment_values))
        )


        for index, (segment, count) in enumerate(
            segment_values
        ):

            percentage = (
                count / len(df) * 100
                if len(df) > 0
                else 0
            )


            with segment_cols[index]:

                st.markdown(
                    f"""
                    <div class="segment-card">

                    <div class="segment-number">
                    {count:,}
                    </div>

                    <div class="segment-label">
                    {get_segment_name(segment)}
                    </div>

                    <br>

                    <div>
                    {percentage:.1f}% of buyers
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # -----------------------------------------------------
    # NUMERIC FEATURE ANALYSIS
    # -----------------------------------------------------

    if numeric_columns:

        st.markdown(
            '<div class="section-title">📈 Feature Analysis</div>',
            unsafe_allow_html=True
        )


        selected_feature = st.selectbox(
            "Select a feature to analyze:",
            numeric_columns
        )


        feature_col1, feature_col2 = st.columns(2)


        # -------------------------------------------------
        # HISTOGRAM
        # -------------------------------------------------

        with feature_col1:

            fig_hist = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                title=f"{selected_feature} Distribution"
            )

            fig_hist.update_layout(
                height=400,
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )


        # -------------------------------------------------
        # BOX PLOT
        # -------------------------------------------------

        with feature_col2:

            if segment_col:

                fig_box = px.box(
                    df,
                    x=segment_col,
                    y=selected_feature,
                    title=f"{selected_feature} by Buyer Segment"
                )

                fig_box.update_layout(
                    height=400,
                    plot_bgcolor="white"
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True
                )


        # -------------------------------------------------
        # SCATTER PLOT
        # -------------------------------------------------

        if len(numeric_columns) >= 2:

            st.markdown(
                '<div class="section-title">🔎 Buyer Feature Relationship</div>',
                unsafe_allow_html=True
            )


            scatter_col1, scatter_col2 = st.columns(2)


            with scatter_col1:

                x_feature = st.selectbox(
                    "X-axis:",
                    numeric_columns,
                    key="x_feature"
                )


            with scatter_col2:

                y_feature = st.selectbox(
                    "Y-axis:",
                    numeric_columns,
                    index=(
                        1
                        if len(numeric_columns) > 1
                        else 0
                    ),
                    key="y_feature"
                )


            if segment_col:

                fig_scatter = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    color=segment_col,
                    title=f"{x_feature} vs {y_feature}",
                    hover_data=df.columns.tolist()
                )

            else:

                fig_scatter = px.scatter(
                    df,
                    x=x_feature,
                    y=y_feature,
                    title=f"{x_feature} vs {y_feature}"
                )


            fig_scatter.update_layout(
                height=500,
                plot_bgcolor="white"
            )


            st.plotly_chart(
                fig_scatter,
                use_container_width=True
            )


        # -------------------------------------------------
        # CORRELATION HEATMAP
        # -------------------------------------------------

        if len(numeric_columns) >= 2:

            st.markdown(
                '<div class="section-title">🔥 Feature Correlation</div>',
                unsafe_allow_html=True
            )


            correlation = (
                df[numeric_columns]
                .corr()
            )


            fig_heatmap = go.Figure(
                data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.columns,
                    text=correlation.round(2).values,
                    texttemplate="%{text}",
                    hoverongaps=False
                )
            )


            fig_heatmap.update_layout(
                title="Correlation Between Numerical Features",
                height=600
            )


            st.plotly_chart(
                fig_heatmap,
                use_container_width=True
            )


# =========================================================
# BUYER SEGMENTS
# =========================================================

elif page == "🎯 Buyer Segments":

    st.markdown(
        '<div class="main-title">🎯 Buyer Segments</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the buyer groups created by K-Means clustering'
        '</div>',
        unsafe_allow_html=True
    )


    if not segment_col:

        st.warning(
            "No buyer segment column was found."
        )

        st.stop()


    # -----------------------------------------------------
    # SEGMENT COUNTS
    # -----------------------------------------------------

    segment_counts = (
        df[segment_col]
        .value_counts()
        .sort_index()
    )


    segment_table = pd.DataFrame({

        "K-Means Segment": [
            get_segment_name(x)
            for x in segment_counts.index
        ],

        "Number of Buyers":
            segment_counts.values,

        "Percentage": [
            f"{x / len(df) * 100:.2f}%"
            for x in segment_counts.values
        ]

    })


    st.subheader(
        "📋 Segment Distribution"
    )


    st.dataframe(
        segment_table,
        hide_index=True,
        use_container_width=True
    )


    # -----------------------------------------------------
    # PIE CHART
    # -----------------------------------------------------

    segment_chart = pd.DataFrame({

        "Segment": [
            get_segment_name(x)
            for x in segment_counts.index
        ],

        "Buyers":
            segment_counts.values

    })


    fig = px.pie(
        segment_chart,
        names="Segment",
        values="Buyers",
        hole=0.4,
        title="Buyer Segment Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # ANALYSIS FILE
    # -----------------------------------------------------

    st.subheader(
        "📊 Segment Characteristics"
    )


    if os.path.exists(ANALYSIS_PATH):

        try:

            analysis = pd.read_csv(
                ANALYSIS_PATH,
                index_col=0
            )


            readable_index = []


            for value in analysis.index:

                try:

                    readable_index.append(
                        get_segment_name(
                            int(value)
                        )
                    )

                except:

                    readable_index.append(
                        str(value)
                    )


            analysis.index = readable_index


            st.dataframe(
                analysis,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                "Unable to load segment analysis."
            )

            st.write(e)


    else:

        st.info(
            "Segment analysis file is not available."
        )


    # -----------------------------------------------------
    # EXPLORE SEGMENT
    # -----------------------------------------------------

    st.subheader(
        "🔎 Explore Individual Segment"
    )


    available_segments = sorted(
        df[segment_col]
        .dropna()
        .unique()
    )


    selected_segment = st.selectbox(
        "Select Segment:",
        available_segments,
        format_func=get_segment_name
    )


    selected_data = df[
        df[segment_col] == selected_segment
    ]


    st.success(
        f"{get_segment_name(selected_segment)} "
        f"contains {len(selected_data):,} buyers."
    )


    # -----------------------------------------------------
    # SEGMENT NUMERIC STATISTICS
    # -----------------------------------------------------

    if numeric_columns:

        statistics = (
            selected_data[numeric_columns]
            .mean()
            .round(2)
            .sort_values(
                ascending=False
            )
        )


        statistics_df = pd.DataFrame({

            "Feature":
                statistics.index,

            "Average Value":
                statistics.values

        })


        st.subheader(
            "📊 Average Feature Values"
        )


        st.dataframe(
            statistics_df,
            hide_index=True,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SEGMENT DATA
    # -----------------------------------------------------

    st.subheader(
        "👥 Buyers in Selected Segment"
    )


    st.dataframe(
        selected_data,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    csv_data = selected_data.to_csv(
        index=False
    )


    st.download_button(
        "📥 Download Selected Segment",
        data=csv_data,
        file_name=f"segment_{selected_segment}.csv",
        mime="text/csv"
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="main-title">🔮 Buyer Segment Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Use the trained K-Means model to classify a new buyer'
        '</div>',
        unsafe_allow_html=True
    )


    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ K-Means model was not found."
        )

        st.code(
            MODEL_PATH
        )

        st.stop()


    try:

        saved_model = joblib.load(
            MODEL_PATH
        )


        if not isinstance(
            saved_model,
            dict
        ):

            st.error(
                "Invalid saved model format."
            )

            st.stop()


        model = saved_model.get(
            "model"
        )

        scaler = saved_model.get(
            "scaler"
        )

        features = saved_model.get(
            "features"
        )


        if (
            model is None
            or scaler is None
            or features is None
        ):

            st.error(
                "The saved model is missing "
                "model, scaler, or features."
            )

            st.stop()


        st.subheader(
            "📝 Enter Buyer Information"
        )


        input_values = {}


        input_cols = st.columns(3)


        for index, feature in enumerate(features):

            # Use median value from dataset
            # when available

            if feature in df.columns:

                default_value = float(
                    df[feature]
                    .median()
                )

            else:

                default_value = 0.0


            with input_cols[index % 3]:

                input_values[feature] = st.number_input(
                    feature,
                    value=default_value,
                    key=f"prediction_{index}"
                )


        st.markdown("---")


        if st.button(
            "🚀 Predict Buyer Segment",
            use_container_width=True
        ):

            input_df = pd.DataFrame(
                [input_values]
            )


            input_df = input_df[
                features
            ]


            input_scaled = scaler.transform(
                input_df
            )


            prediction = model.predict(
                input_scaled
            )[0]


            predicted_segment = (
                int(prediction) + 1
            )


            predicted_name = get_segment_name(
                predicted_segment
            )


            st.subheader(
                "🎯 Prediction Result"
            )


            result_col1, result_col2, result_col3 = st.columns(3)


            with result_col1:

                st.metric(
                    "K-Means Cluster",
                    int(prediction)
                )


            with result_col2:

                st.metric(
                    "Segment",
                    predicted_segment
                )


            with result_col3:

                st.metric(
                    "Buyer Type",
                    predicted_name
                )


            st.success(
                f"🎯 This buyer belongs to **{predicted_name}**."
            )


            st.info(
                f"The K-Means model assigned this buyer "
                f"to cluster {int(prediction)}."
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.write(
            f"Error: {e}"
        )


# =========================================================
# DATASET
# =========================================================

elif page == "📊 Dataset":

    st.markdown(
        '<div class="main-title">📊 Buyer Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the complete segmented real estate buyer dataset'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # DATASET METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Numerical Features",
            len(numeric_columns)
        )


    # -----------------------------------------------------
    # DATASET TABLE
    # -----------------------------------------------------

    st.subheader(
        "📋 Complete Dataset"
    )


    st.dataframe(
        df,
        use_container_width=True,
        height=550
    )


    # -----------------------------------------------------
    # DATA TYPES
    # -----------------------------------------------------

    st.subheader(
        "🔧 Dataset Structure"
    )


    dtype_df = pd.DataFrame({

        "Column":
            df.columns,

        "Data Type":
            [
                str(x)
                for x in df.dtypes
            ],

        "Missing Values":
            [
                int(x)
                for x in df.isnull().sum()
            ]

    })


    st.dataframe(
        dtype_df,
        hide_index=True,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.subheader(
        "📥 Download Dataset"
    )


    complete_csv = df.to_csv(
        index=False
    )


    st.download_button(

        label="⬇️ Download Complete Dataset",

        data=complete_csv,

        file_name="segmented_buyers.csv",

        mime="text/csv",

        use_container_width=True

    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🏠 <b>Real Estate Buyer Intelligence System</b>

    <br>

    Powered by Python • Streamlit • Pandas • Scikit-Learn • K-Means

    </div>
    """,
    unsafe_allow_html=True
)