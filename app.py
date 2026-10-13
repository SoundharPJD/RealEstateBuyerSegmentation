import streamlit as st
import pandas as pd
import os
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
   ===================================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0b1730 45%,
        #111827 100%
    );
    color: #f8fafc;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #081426 0%,
        #101d38 100%
    );
    border-right: 1px solid #243453;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}


/* =====================================================
   MAIN TITLE
   ===================================================== */

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #a78bfa,
        #22d3ee
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.subtitle {
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 25px;
}


/* =====================================================
   GLASS CARDS
   ===================================================== */

.glass-card {

    background: rgba(17, 29, 52, 0.82);

    border: 1px solid rgba(96, 165, 250, 0.20);

    border-radius: 18px;

    padding: 24px;

    margin-bottom: 18px;

    box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.25);

}


/* =====================================================
   METRIC CARDS
   ===================================================== */

.metric-card {

    background: linear-gradient(
        145deg,
        rgba(30, 64, 175, 0.35),
        rgba(88, 28, 135, 0.25)
    );

    border: 1px solid rgba(147, 197, 253, 0.25);

    border-radius: 18px;

    padding: 22px;

    min-height: 125px;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25);

}


.metric-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
}


.metric-value {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}


.metric-icon {
    font-size: 25px;
}


/* =====================================================
   SEGMENT CARDS
   ===================================================== */

.segment-card {

    background: linear-gradient(
        145deg,
        rgba(37, 99, 235, 0.30),
        rgba(124, 58, 237, 0.25)
    );

    border: 1px solid rgba(129, 140, 248, 0.25);

    border-radius: 18px;

    padding: 22px;

    text-align: center;

    transition: 0.3s;

}


.segment-card:hover {

    transform: translateY(-5px);

    border: 1px solid #60a5fa;

    box-shadow:
        0 15px 35px rgba(59, 130, 246, 0.20);

}


.segment-number {

    font-size: 32px;

    font-weight: 800;

    color: #93c5fd;

}


.segment-label {

    color: #cbd5e1;

    font-size: 14px;

}


.segment-count {

    font-size: 22px;

    font-weight: 700;

    color: #f8fafc;

    margin-top: 8px;

}


/* =====================================================
   SECTION HEADINGS
   ===================================================== */

.section-title {

    font-size: 24px;

    font-weight: 750;

    color: #f8fafc;

    margin-top: 25px;

    margin-bottom: 15px;

}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 12px 20px;

    font-weight: 700;

    transition: 0.3s;

}


.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 8px 25px rgba(99, 102, 241, 0.40);

}


/* =====================================================
   DOWNLOAD BUTTON
   ===================================================== */

.stDownloadButton > button {

    background: #172554;

    color: #bfdbfe;

    border: 1px solid #3b82f6;

    border-radius: 10px;

}


/* =====================================================
   INPUT BOXES
   ===================================================== */

div[data-baseweb="input"] {

    background-color: #111c31;

    border-radius: 10px;

}


input {

    color: white !important;

}


/* =====================================================
   SELECTBOX
   ===================================================== */

div[data-baseweb="select"] {

    background-color: #111c31;

    border-radius: 10px;

}


/* =====================================================
   DATAFRAME
   ===================================================== */

[data-testid="stDataFrame"] {

    border-radius: 15px;

    overflow: hidden;

}


/* =====================================================
   INFO BOX
   ===================================================== */

.info-box {

    background: rgba(30, 41, 59, 0.8);

    border-left: 4px solid #60a5fa;

    padding: 16px;

    border-radius: 10px;

    color: #cbd5e1;

}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {

    text-align: center;

    color: #64748b;

    padding: 30px;

    margin-top: 50px;

    border-top: 1px solid #1e293b;

}

</style>
""", unsafe_allow_html=True)


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

    except Exception:
        return str(segment)


# =========================================================
# CHECK DATA
# =========================================================

if not os.path.isfile(DATA_PATH):

    st.error("❌ Buyer dataset was not found.")

    st.write("Expected file:")

    st.code(DATA_PATH)

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error("Unable to load dataset.")

    st.code(str(e))

    st.stop()
# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div style="text-align:center;">

    <div style="font-size:55px;">🏠</div>

    <h2 style="margin-top:-10px;">
    REAL ESTATE AI
    </h2>

    <p style="color:#94a3b8;">
    Buyer Intelligence Platform
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown("---")


page = st.sidebar.radio(

    "MAIN MENU",

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
    <div class="info-box">

    🤖 <b>Machine Learning</b><br>
    K-Means Clustering

    <br><br>

    📊 <b>Analytics</b><br>
    Buyer segmentation

    <br><br>

    🎯 <b>Clusters</b><br>
    3 buyer groups

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CHECK DATA
# =========================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "❌ Segmented dataset was not found."
    )

    st.code(
        "data/processed/segmented_buyers.csv"
    )

    st.info(
        "Run your model training program first."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)


# =========================================================
# FIND SEGMENT COLUMN
# =========================================================

segment_columns = [

    "buyer_segment",

    "segment",

    "Segment",

    "cluster",

    "Cluster"

]


segment_col = None


for column in segment_columns:

    if column in df.columns:

        segment_col = column

        break


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Real Estate Buyer Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered buyer segmentation using K-Means clustering'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # TOP STATUS
    # =====================================================

    st.markdown(
        """
        <div class="glass-card">

        🟢 <b>System Status:</b> Online
        &nbsp;&nbsp;&nbsp;

        🤖 <b>Model:</b> K-Means
        &nbsp;&nbsp;&nbsp;

        🎯 <b>Clusters:</b> 3
        &nbsp;&nbsp;&nbsp;

        📁 <b>Dataset:</b> Loaded

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📌 Key Statistics'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-icon">👥</div>

            <div class="metric-title">
            TOTAL BUYERS
            </div>

            <div class="metric-value">
            {len(df):,}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        segments = (
            df[segment_col].nunique()
            if segment_col
            else 0
        )

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-icon">🎯</div>

            <div class="metric-title">
            BUYER SEGMENTS
            </div>

            <div class="metric-value">
            {segments}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-icon">📊</div>

            <div class="metric-title">
            DATASET COLUMNS
            </div>

            <div class="metric-value">
            {len(df.columns)}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            """
            <div class="metric-card">

            <div class="metric-icon">🤖</div>

            <div class="metric-title">
            ML ALGORITHM
            </div>

            <div class="metric-value"
            style="font-size:22px;">
            K-Means
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # SEGMENT CARDS
    # =====================================================

    if segment_col:

        st.markdown(
            '<div class="section-title">'
            '🎯 Buyer Segment Overview'
            '</div>',
            unsafe_allow_html=True
        )


        counts = (
            df[segment_col]
            .value_counts()
            .sort_index()
        )


        segment_cols = st.columns(
            len(counts)
        )


        for i, (segment, count) in enumerate(
            counts.items()
        ):

            with segment_cols[i]:

                st.markdown(
                    f"""
                    <div class="segment-card">

                    <div class="segment-number">
                    {int(segment)}
                    </div>

                    <div class="segment-label">
                    K-MEANS SEGMENT
                    </div>

                    <div class="segment-count">
                    {count:,} Buyers
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # =====================================================
    # CHARTS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Buyer Distribution'
        '</div>',
        unsafe_allow_html=True
    )


    chart_col1, chart_col2 = st.columns(2)


    if segment_col:

        with chart_col1:

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            st.write(
                "### 🎯 Segment Distribution"
            )

            chart_data = (
                df[segment_col]
                .value_counts()
                .sort_index()
            )

            chart_data.index = [

                get_segment_name(x)

                for x in chart_data.index

            ]

            st.bar_chart(
                chart_data
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with chart_col2:

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            st.write(
                "### 📊 Segment Percentage"
            )

            percentage = (

                df[segment_col]
                .value_counts(
                    normalize=True
                )
                .sort_index()
                * 100

            ).round(2)

            percentage.index = [

                get_segment_name(x)

                for x in percentage.index

            ]

            st.bar_chart(
                percentage
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# =========================================================
# BUYER SEGMENTS
# =========================================================

elif page == "🎯 Buyer Segments":

    st.markdown(
        '<div class="main-title">'
        'Buyer Segments'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the characteristics of each K-Means cluster'
        '</div>',
        unsafe_allow_html=True
    )


    if not segment_col:

        st.error(
            "No buyer segment column found."
        )

        st.stop()


    counts = (
        df[segment_col]
        .value_counts()
        .sort_index()
    )


    # =====================================================
    # SEGMENT SELECTOR
    # =====================================================

    selected_segment = st.selectbox(

        "🔎 Select Buyer Segment",

        sorted(
            df[segment_col]
            .dropna()
            .unique()
        ),

        format_func=get_segment_name

    )


    selected_data = df[
        df[segment_col] == selected_segment
    ]


    # =====================================================
    # SELECTED SEGMENT HEADER
    # =====================================================

    st.markdown(
        f"""
        <div class="glass-card">

        <h2>
        🏷️ {get_segment_name(selected_segment)}
        </h2>

        <p style="color:#94a3b8;">
        Buyers belonging to this K-Means cluster
        </p>

        <h1>
        {len(selected_data):,}
        </h1>

        <p>
        Buyers in this segment
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # STATISTICS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Segment Characteristics'
        '</div>',
        unsafe_allow_html=True
    )


    numeric_columns = (

        selected_data
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()

    )


    numeric_columns = [

        col

        for col in numeric_columns

        if col != segment_col

    ]


    if numeric_columns:

        stats = (

            selected_data[
                numeric_columns
            ]
            .mean()
            .round(2)

        )


        stats_df = pd.DataFrame({

            "Feature":
                stats.index,

            "Average":
                stats.values

        })


        st.dataframe(

            stats_df,

            hide_index=True,

            width="stretch"

        )


    # =====================================================
    # SEGMENT DATA
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '👥 Buyers in This Segment'
        '</div>',
        unsafe_allow_html=True
    )


    st.dataframe(

        selected_data,

        width="stretch",

        height=400

    )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        "📥 Download This Segment",

        data=selected_data.to_csv(
            index=False
        ),

        file_name=(
            f"segment_{selected_segment}.csv"
        ),

        mime="text/csv"

    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="main-title">'
        'AI Buyer Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter buyer information and let K-Means '
        'identify the buyer segment'
        '</div>',
        unsafe_allow_html=True
    )


    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ K-Means model was not found."
        )

        st.code(
            "models/buyer_segmentation_model.pkl"
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
                "Invalid model format."
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
                "Model, scaler or features are missing."
            )

            st.stop()


        # =====================================================
        # FORM
        # =====================================================

        st.markdown(
            """
            <div class="glass-card">

            <h3>
            📝 Enter Buyer Information
            </h3>

            <p style="color:#94a3b8;">
            Enter values for the features used during
            K-Means training.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        input_values = {}


        input_columns = st.columns(3)


        for index, feature in enumerate(
            features
        ):

            column = input_columns[
                index % 3
            ]


            with column:

                input_values[feature] = (

                    st.number_input(

                        feature,

                        value=0.0,

                        key=f"feature_{index}"

                    )

                )


        st.markdown("")


        # =====================================================
        # PREDICT
        # =====================================================

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


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                f"""
                <div class="glass-card"
                style="text-align:center;">

                <div style="font-size:60px;">
                🎯
                </div>

                <h2>
                Prediction Complete
                </h2>

                <p style="color:#94a3b8;">
                K-Means identified this buyer as
                </p>

                <h1 style="
                color:#93c5fd;
                font-size:38px;
                ">

                {predicted_name}

                </h1>

                <p>
                K-Means Cluster: {prediction}
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


            result_df = pd.DataFrame({

                "K-Means Cluster": [
                    prediction
                ],

                "Buyer Segment": [
                    predicted_segment
                ],

                "Segment Name": [
                    predicted_name
                ]

            })


            st.dataframe(

                result_df,

                hide_index=True,

                width="stretch"

            )


    except Exception as e:

        st.error(
            "❌ Error loading prediction model."
        )

        st.code(
            str(e)
        )


# =========================================================
# DATASET
# =========================================================

elif page == "📊 Dataset":

    st.markdown(
        '<div class="main-title">'
        'Dataset Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore and download the complete segmented dataset'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )


    with c2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with c3:

        st.metric(
            "Segments",
            (
                df[segment_col].nunique()
                if segment_col
                else 0
            )
        )


    # =====================================================
    # SEARCH
    # =====================================================

    search = st.text_input(
        "🔎 Search dataset"
    )


    display_df = df.copy()


    if search:

        mask = display_df.astype(
            str
        ).apply(

            lambda row:
            row.str.contains(
                search,
                case=False,
                na=False
            ).any(),

            axis=1

        )

        display_df = display_df[
            mask
        ]


    # =====================================================
    # DISPLAY
    # =====================================================

    st.dataframe(

        display_df,

        width="stretch",

        height=500

    )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(

        "📥 Download Complete Dataset",

        data=df.to_csv(
            index=False
        ),

        file_name="segmented_buyers.csv",

        mime="text/csv"

    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🏠 <b>Real Estate Buyer Intelligence Platform</b>

    <br><br>

    Powered by 🤖 K-Means Machine Learning
    &nbsp; • &nbsp;
    Built with Streamlit

    </div>
    """,
    unsafe_allow_html=True
)