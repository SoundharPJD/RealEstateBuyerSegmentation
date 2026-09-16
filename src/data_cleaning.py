import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    clients_file = RAW_DATA_DIR / "clients.csv"
    properties_file = RAW_DATA_DIR / "properties.csv"

    clients = pd.read_csv(clients_file)
    properties = pd.read_csv(properties_file)

    return clients, properties


# ---------------------------------------------------------
# CLEAN COLUMN NAMES
# ---------------------------------------------------------

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

def check_columns(clients, properties):

    print("\nChecking required columns...")

    if "client_id" not in clients.columns:
        raise ValueError("client_id column not found in clients.csv")

    if "client_ref" not in properties.columns:
        raise ValueError("client_ref column not found in properties.csv")

    print("Required columns found.")


# ---------------------------------------------------------
# CREATE PROPERTY SUMMARY FOR EACH CLIENT
# ---------------------------------------------------------

def create_property_summary(properties):

    print("\nCreating property summary...")

    # Number of properties associated with each client
    property_summary = (
        properties
        .groupby("client_ref")
        .agg(
            property_count=("client_ref", "count")
        )
        .reset_index()
    )

    # Rename client_ref so it matches clients.csv
    property_summary = property_summary.rename(
        columns={"client_ref": "client_id"}
    )

    return property_summary


# ---------------------------------------------------------
# MERGE CLIENT + PROPERTY DATA
# ---------------------------------------------------------

def merge_data(clients, property_summary):

    print("\nMerging client and property data...")

    merged_data = clients.merge(
        property_summary,
        on="client_id",
        how="left"
    )

    # Clients without properties get 0
    merged_data["property_count"] = (
        merged_data["property_count"]
        .fillna(0)
        .astype(int)
    )

    return merged_data


# ---------------------------------------------------------
# SAVE PROCESSED DATA
# ---------------------------------------------------------

def save_data(df):

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = PROCESSED_DATA_DIR / "merged_data.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\nMerged dataset saved successfully!")
    print(f"Location: {output_file}")


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("REAL ESTATE BUYER SEGMENTATION")
    print("=" * 60)

    # Load
    clients, properties = load_data()

    print("\nClients dataset:")
    print(clients.shape)

    print("\nProperties dataset:")
    print(properties.shape)

    # Clean column names
    clients = clean_column_names(clients)
    properties = clean_column_names(properties)

    # Check columns
    check_columns(clients, properties)

    # Create property summary
    property_summary = create_property_summary(properties)

    print("\nProperty summary:")
    print(property_summary.head())

    # Merge datasets
    merged_data = merge_data(
        clients,
        property_summary
    )

    print("\nMerged dataset:")
    print(merged_data.shape)

    print("\nMerged columns:")
    print(merged_data.columns.tolist())

    print("\nFirst 5 rows:")
    print(merged_data.head())

    # Save
    save_data(merged_data)

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETED")
    print("=" * 60)


# ---------------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------------
def save_data(df):
    output_path = "data/processed/merged_data.csv"

    df.to_csv(output_path, index=False)

    print("\nMerged dataset saved successfully!")
    print(f"Location: {output_path}")


if __name__ == "__main__":
    main()