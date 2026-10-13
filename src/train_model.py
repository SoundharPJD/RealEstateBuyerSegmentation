import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# STEP 5 - BUYER SEGMENTATION
# ---------------------------------------------------------

input_file = "data/processed/ml_ready_data.csv"

output_file = ""segmented_buyers.csv""

model_file = "models/buyer_segmentation_model.pkl"

analysis_file = "data/processed/segment_analysis.csv"


print("Loading ML-ready dataset...")

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# SELECT NUMERICAL COLUMNS
# ---------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


# ---------------------------------------------------------
# REMOVE ID COLUMNS
# ---------------------------------------------------------

id_columns = [
    column
    for column in numeric_columns
    if "id" in column.lower()
]


features = [
    column
    for column in numeric_columns
    if column not in id_columns
]


print("\nFeatures used for segmentation:")

for column in features:

    print("-", column)


# ---------------------------------------------------------
# CREATE FEATURE DATASET
# ---------------------------------------------------------

X = df[features].copy()


# ---------------------------------------------------------
# SCALE THE DATA
# ---------------------------------------------------------

print("\nScaling data...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# CREATE K-MEANS MODEL
# ---------------------------------------------------------

print("\nTraining K-Means model...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

kmeans.fit(X_scaled)


# ---------------------------------------------------------
# ADD SEGMENT NUMBER
# ---------------------------------------------------------

df["buyer_segment"] = kmeans.labels_ + 1


# ---------------------------------------------------------
# SAVE SEGMENTED DATASET
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    {
        "model": kmeans,
        "scaler": scaler,
        "features": features
    },
    model_file
)


# ---------------------------------------------------------
# DISPLAY SEGMENT COUNTS
# ---------------------------------------------------------

print("\nBuyer segmentation completed successfully!")

print("\nNumber of buyers in each segment:")

print(
    df["buyer_segment"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# SAVE OUTPUT FILE INFORMATION
# ---------------------------------------------------------

print("\nSegmented dataset saved to:")

print(output_file)


print("\nModel saved to:")

print(model_file)


# =========================================================
# STEP 6 - BUYER SEGMENT ANALYSIS
# =========================================================

print("\n========================================")

print("BUYER SEGMENT ANALYSIS")

print("========================================")


# ---------------------------------------------------------
# CALCULATE AVERAGE VALUES
# ---------------------------------------------------------

segment_analysis = (

    df.groupby("buyer_segment")[features]
    .mean()
    .round(2)

)


# ---------------------------------------------------------
# DISPLAY ANALYSIS
# ---------------------------------------------------------

print("\nAverage values for each buyer segment:")

print(segment_analysis)


# ---------------------------------------------------------
# SAVE SEGMENT ANALYSIS
# ---------------------------------------------------------

segment_analysis.to_csv(
    analysis_file
)


print("\nSegment analysis saved to:")

print(analysis_file)


# ---------------------------------------------------------
# DISPLAY EACH SEGMENT
# ---------------------------------------------------------

print("\n========================================")

print("SEGMENT DETAILS")

print("========================================")


for segment in sorted(
    df["buyer_segment"].unique()
):

    print(
        f"\nSegment {segment}"
    )

    print(
        "Number of buyers:",
        len(
            df[
                df["buyer_segment"] == segment
            ]
        )
    )

    print(
        "Average values:"
    )

    print(
        segment_analysis.loc[segment]
    )


print("\n========================================")

print("MODEL TRAINING COMPLETED")

print("========================================")