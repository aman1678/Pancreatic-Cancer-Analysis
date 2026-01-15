# Utility function for model evaluation and result printing
def print_results(results: dict):
    print("Model Performance (ROC-AUC)")
    print("-" * 30)
    for model, score in results.items():
        print(f"{model}: {score:.4f}")
