import pandas as pd


# ---------------------------------------------------------
# STEP 4 - PREPARE DATA FOR MACHINE LEARNING
# ---------------------------------------------------------

input_file = "data/processed/cleaned_data.csv"
output_file = "data/processed/ml_ready_data.csv"

print("Loading cleaned dataset...")

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

print("\nColumns in dataset:")
print(df.columns.tolist())


# ---------------------------------------------------------
# Convert text columns into numbers
# ---------------------------------------------------------

text_columns = df.select_dtypes(
    include=["object"]
).columns

print("\nText columns:")
print(text_columns.tolist())


df_encoded = pd.get_dummies(
    df,
    columns=text_columns,
    drop_first=True
)


# ---------------------------------------------------------
# Convert True/False into 0/1
# ---------------------------------------------------------

df_encoded = df_encoded.astype(int)


# ---------------------------------------------------------
# Save ML-ready dataset
# ---------------------------------------------------------

df_encoded.to_csv(
    output_file,
    index=False
)

print("\nML preparation completed successfully!")

print("ML-ready shape:", df_encoded.shape)

print("Saved to:", output_file)