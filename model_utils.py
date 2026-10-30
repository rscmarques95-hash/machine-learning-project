"""
model_utils.py
--------------
Helper functions to evaluate and compare classification models.

How to use it in a notebook:
    from model_utils import evaluate_model, plot_confusion_matrix
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)


def evaluate_model(name, model, X_test, y_test):
    """
    Make predictions on the test set and return the main metrics in a dictionary.

    - accuracy:  % of all predictions that are correct
    - precision: of the animals we said 'adopted', % that really were adopted
    - recall:    of the animals that were really adopted, % that we found
    - f1:        a balance between precision and recall
    """
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
    }


def plot_confusion_matrix(name, model, X_test, y_test):
    """Show the confusion matrix: correct and wrong predictions for each class."""
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Not adopted", "Adopted"],
        cmap="Blues",
    )
    plt.title(f"Confusion matrix – {name}")
    plt.grid(False)
    plt.show()


def results_table(results):
    """Turn a list of metric dictionaries into a table, best F1 first."""
    return pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
