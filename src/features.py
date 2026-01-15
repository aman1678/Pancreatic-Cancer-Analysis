import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Function to build and encode features
def build_features(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    # Transform X
    X_processed = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=numerical_cols + 
                list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )

    return X_processed, y
