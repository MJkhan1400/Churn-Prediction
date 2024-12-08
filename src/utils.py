import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names):
    """
    Plots feature importance for a given model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title("Feature Importance")
    plt.show()

