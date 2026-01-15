import pandas as pd


TARGET_COL = "Survival_Status"

# Function to load and validate data
def load_and_validate(path: str) -> pd.DataFrame:

    df = None

    #Compatible with both .csv and .xlsx files
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    # Basic validation
    assert df[TARGET_COL].isnull().sum() == 0, "Target contains nulls"
    assert df.duplicated().sum() == 0, "Duplicate rows detected"

    # Deal with null values by changing them to appropriate defaults
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
    for col in df.select_dtypes(exclude=["object"]).columns:
        df[col] = df[col].fillna(0)

    return df
