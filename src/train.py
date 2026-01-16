from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from typing import Dict, Tuple
import pandas as pd


def train_models(X, y) -> Tuple[Dict[str, BaseEstimator], Dict[str, float], pd.DataFrame, pd.Series, Dict[str, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )
    
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,  # Increased for convergence
            class_weight="balanced",
            random_state=42,
            solver='lbfgs'  # Explicit solver
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,  # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",  # Handle class imbalance
            random_state=42
        )
    }

    results = {}
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        results[name] = cv_scores.mean()
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        trained_models[name] = model
        
    return trained_models, results, X_test, y_test, predictions