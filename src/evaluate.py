import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, classification_report


# ROC Curve
def plot_roc(y_true, y_prob, label=None):
    """
    Plot ROC curve with optional label.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=label)
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()


# Precision-Recall Curve
def plot_precision_recall(y_true, y_prob):
    """
    Plot precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix with annotations.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


# Classification Report
def print_classification_report(y_true, y_pred, model_name):
    """
    Print standard classification report.
    """
    report = classification_report(y_true, y_pred)
    print(f"=== Classification Report: {model_name} ===\n{report}")


# Feature Importance Plot
def plot_feature_importance(model, X, model_name, top_n=10):
    """
    Plot top N feature importances, aggregating one-hot encoded features by base variable.
    Works only for models with 'feature_importances_' attribute.
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not have feature importances.")
        return

    # Create DataFrame of feature importances
    df_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })

    # Aggregate one-hot encoded features by base name
    df_imp['base_feature'] = df_imp['feature'].str.split('_').str[0]
    df_agg = df_imp.groupby('base_feature')['importance'].max().sort_values(ascending=False)

    # Take top N features
    df_top = df_agg.head(top_n)

    plt.figure(figsize=(8,6))
    df_top[::-1].plot(kind='barh', color='skyblue')  # reverse for top-down
    plt.title(f'Top {top_n} Feature Importances: {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
