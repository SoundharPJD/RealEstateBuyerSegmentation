import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# STEP 7 - VISUALIZE BUYER SEGMENTS
# ---------------------------------------------------------

input_file = "data/processed/segmented_buyers.csv"

df = pd.read_csv(input_file)

print("Segmented dataset loaded successfully!")

# ---------------------------------------------------------
# Create plots folder
# ---------------------------------------------------------

os.makedirs("outputs/plots", exist_ok=True)

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
    print("ERROR: No segment column found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

# ---------------------------------------------------------
# 1. Segment Count Chart
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=segment_col
)

plt.title("Number of Buyers in Each Segment")
plt.xlabel("Buyer Segment")
plt.ylabel("Number of Buyers")

plt.tight_layout()
plt.savefig("outputs/plots/segment_counts.png")
plt.show()

# ---------------------------------------------------------
# 2. Age vs Segment
# ---------------------------------------------------------

if "age" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=segment_col,
        y="age"
    )

    plt.title("Age Distribution by Buyer Segment")
    plt.xlabel("Buyer Segment")
    plt.ylabel("Age")

    plt.tight_layout()
    plt.savefig("outputs/plots/age_by_segment.png")
    plt.show()

# --------------