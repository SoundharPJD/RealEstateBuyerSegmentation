import pandas as pd

# ---------------------------------------------------------
# STEP 12 - UNDERSTAND BUYER SEGMENTS
# ---------------------------------------------------------

input_file = "data/processed/segmented_buyers.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("BUYER SEGMENT ANALYSIS")
print("=" * 60)

# Find the segment column
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

if segment_col is None:
    print("ERROR: Segment column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

print("\nSegment column:", segment_col)

# ---------------------------------------------------------
# Number of buyers in each segment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMBER OF BUYERS")
print("=" * 60)

counts = df[segment_col].value_counts().sort_index()

print(counts)

# ---------------------------------------------------------
# Average numerical values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("AVERAGE VALUES BY SEGMENT")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

# Remove segment column
numeric_columns = [
    column
    for column in numeric_columns
    if column != segment_col
]

summary = df.groupby(
    segment_col
)[numeric_columns].mean()

print("\n")
print(summary.round(2))

# ---------------------------------------------------------
# Save summary
# ---------------------------------------------------------

output_file = "outputs/segment_summary.csv"

summary.round(2).to_csv(output_file)

print("\n" + "=" * 60)
print("SUMMARY SAVED")
print("=" * 60)

print(output_file)