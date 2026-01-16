import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def build_features(df: pd.DataFrame, target_col: str = "Survival_Status") -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col, "Country"])
    y = df[target_col]

    # REMOVE SURVIVAL_TIME_MONTHS - this is data leakage!
    if "Survival_Time_Months" in X.columns:
        X = X.drop(columns=["Survival_Time_Months"])
    
    # Only Age is numerical now
    numerical_cols = ["Age"]
    
    one_hot_cols = ["Gender", "Urban_vs_Rural", "Treatment_Type"]
    
    # Define ordinal categories with explicit ordering
    ordinal_cols_config = {
        "Stage_at_Diagnosis": ["Stage I", "Stage II", "Stage III", "Stage IV"],
        "Physical_Activity_Level": ["Low", "Medium", "High"],
        "Diet_Processed_Food": ["Low", "Medium", "High"],
        "Access_to_Healthcare": ["Low", "Medium", "High"],
        "Economic_Status": ["Low", "Middle", "High"]
    }
    
    binary_cols = [
        "Smoking_History", "Obesity", "Diabetes", "Chronic_Pancreatitis",
        "Family_History", "Hereditary_Condition", "Jaundice", 
        "Abdominal_Discomfort", "Back_Pain", "Weight_Loss",
        "Development_of_Type2_Diabetes", "Alcohol_Consumption"
    ]
    
    # Create ordinal encoders with explicit ordering
    ordinal_transformers = [
        (f"ord_{col}", OrdinalEncoder(categories=[cats]), [col])
        for col, cats in ordinal_cols_config.items()
    ]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            *ordinal_transformers,
            ("oneCat", OneHotEncoder(drop="first", sparse_output=False), one_hot_cols),
            ("passed", "passthrough", binary_cols)
        ]    
    )
    
    # Transform X
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    ordinal_cols = list(ordinal_cols_config.keys())
    feature_names = (
        numerical_cols + 
        ordinal_cols +
        list(preprocessor.named_transformers_["oneCat"].get_feature_names_out(one_hot_cols)) + 
        binary_cols
    )
    
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_processed, y