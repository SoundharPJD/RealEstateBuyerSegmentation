import pandas as pd


# ---------------------------------------------------------
# STEP 3 - CLEAN DATA
# ---------------------------------------------------------

def clean_data():

    # Location of merged dataset
    input_file = "data/processed/merged_data.csv"

    # Location for cleaned dataset
    output_file = "data/processed/cleaned_data.csv"

    print("Loading merged dataset...")

    # Read dataset
    df = pd.read_csv(input_file)

    print("Original shape:", df.shape)

    # -----------------------------------------------------
    # 1. Remove duplicate rows
    # -----------------------------------------------------

    df = df.drop_duplicates()

    print("After removing duplicates:", df.shape)

    # -----------------------------------------------------
    # 2. Remove completely empty rows
    # -----------------------------------------------------

    df = df.dropna(how="all")

    # -----------------------------------------------------
    # 3. Clean column names
    # -----------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # -----------------------------------------------------
    # 4. Handle missing values
    # -----------------------------------------------------

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].fillna("Unknown")

        else:

            df[column] = df[column].fillna(
                df[column].median()
            )

    # -----------------------------------------------------
    # 5. Remove extra spaces from text
    # -----------------------------------------------------

    for column in df.select_dtypes(include="object").columns:

        df[column] = df[column].astype(str).str.strip()

    # -----------------------------------------------------
    # 6. Save cleaned dataset
    # -----------------------------------------------------

    df.to_csv(output_file, index=False)

    print("\nCleaning completed successfully!")

    print("Cleaned shape:", df.shape)

    print("Saved to:", output_file)


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":
    clean_data()