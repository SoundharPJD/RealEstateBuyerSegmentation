import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# STEP 7 - VISUALIZE BUYER SEGMENTS
# ---------------------------------------------------------

input_file = "data/processed/segmented_buyers.csv"

df = pd.read_csv(input_file)

print("Segmented dataset loaded successfully!")


# ---------------------------------------------------------
# Create plots folder
# ---------------------------------------------------------

import os

os.makedirs("outputs/plots", exist_ok=True)


# ---------------------------------------------------------
# 1. Segment Count Chart
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="buyer_segment"
)

plt.title("Number of Buyers in Each Segment")
plt.xlabel("Buyer Segment")
plt.ylabel("Number of Buyers")

plt.tight_layout()

plt.savefig(
    "outputs/plots/segment_counts.png"
)

plt.show()


# ---------------------------------------------------------
# 2. Age vs Segment
# ---------------------------------------------------------

if "age" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="buyer_segment",
        y="age"
    )

    plt.title("Age Distribution by Buyer Segment")
    plt.xlabel("Buyer Segment")
    plt.ylabel("Age")

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/age_by_segment.png"
    )

    plt.show()


# ---------------------------------------------------------
# 3. Property Count vs Segment
# ---------------------------------------------------------

if "property_count" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="buyer_segment",
        y="property_count"
    )

    plt.title("Property Count by Buyer Segment")
    plt.xlabel("Buyer Segment")
    plt.ylabel("Number of Properties")

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/property_count_by_segment.png"
    )

    plt.show()


print("\nVisualization completed successfully!")

print("\nPlots saved inside:")
print("outputs/plots")