from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from typing import Dict, Tuple
import pandas as pd
import os
from data_validation import load_and_validate
from features import build_features
from evaluate import plot_roc, plot_precision_recall, plot_confusion_matrix, print_classification_report, plot_feature_importance, plot_permutation_importance
from utils import print_results

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
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "svm": SVC(
            kernel='rbf',
            C=1.0,
            probability=True,  # For ROC curves
            class_weight='balanced',
            random_state=42
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform'
        ),
        "naive_bayes": GaussianNB()
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


def main():
    # Load and validate data
    data_path = os.path.join("..", "data", "pancreatic_cancer_prediction_sample.csv")
    df = load_and_validate(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Build features
    X, y = build_features(df)
    print(f"Features built. X shape: {X.shape}, y shape: {y.shape}")

    # Train models
    trained_models, results, X_test, y_test, predictions = train_models(X, y)

    # Print results
    print_results(results)

    # Evaluate models
    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        preds = predictions[name]
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        if probs is not None:
            plot_roc(y_test, probs, label=name)
            plot_precision_recall(y_test, probs)

        plot_confusion_matrix(y_test, preds, name)
        print_classification_report(y_test, preds, name)

        # Feature importance for tree-based models
        if name in ["random_forest", "gradient_boosting"]:
            plot_feature_importance(model, X, name)
        
        # Permutation importance for all models
        plot_permutation_importance(model, X_test, y_test, name)


if __name__ == "__main__":
    main()


