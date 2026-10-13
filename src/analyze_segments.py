import pandas as pd
import os

# ---------------------------------------------------------
# STEP 12 - UNDERSTAND BUYER SEGMENTS
# ---------------------------------------------------------

input_file = "data/processed/segmented_buyers.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("BUYER SEGMENT ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# Find Segment Column
# ---------------------------------------------------------

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

print("\nSegment Column:", segment_col)

# ---------------------------------------------------------
# Number of Buyers in Each Segment
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMBER OF BUYERS")
print("=" * 60)

counts