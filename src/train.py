from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from typing import Dict, Tuple
import pandas as pd

# Train different models and evaluate their performance
def train_models(X, y) -> Tuple[Dict[str, BaseEstimator], Dict[str, float], pd.DataFrame, pd.Series, Dict[str, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Can add more models if desired
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
    }

    results = {}
    trained_models = {}
    predictions = {}

    # Train and evaluate each model with cross-validated ROC-AUC
    for name, model in models.items():
        # Use cross-validation for more robust evaluation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        results[name] = cv_scores.mean()  # Store mean CV AUC

        # Still fit on full train set for predictions on test set
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        trained_models[name] = model

        
    return trained_models, results, X_test, y_test, predictions
