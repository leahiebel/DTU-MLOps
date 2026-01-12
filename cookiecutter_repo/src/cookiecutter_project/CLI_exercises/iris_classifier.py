"""
Breast Cancer Classifier CLI

This module demonstrates advanced Typer features for building command-line interfaces (CLI).
It provides a machine learning workflow to train and evaluate classifiers on the breast cancer dataset.

The Typer framework is used to create a hierarchical CLI structure with:
- Main `app`: The root CLI application
- `train` subcommand: A command group with multiple training options (SVM and KNN)
- `evaluate` command: Evaluates a trained model

Key Typer features demonstrated:
- Creating nested CLI commands using `app.add_typer()`
- Type annotations for automatic validation and help generation
- Both options (with flags) and arguments in CLI functions

Usage examples:
  python script.py train svm --kernel rbf --output my_svm.pkl
  python script.py train knn --n-neighbors 3 --output my_knn.pkl
  python script.py evaluate my_svm.pkl
"""

import pickle
from typing import Annotated

import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Initialize the main Typer application
app = typer.Typer()

# Create a nested Typer app for training commands
# This allows grouping multiple training algorithms under a "train" subcommand
train_app = typer.Typer()
app.add_typer(train_app, name="train")

# Load and prepare the breast cancer dataset
# This is done at module load time so the data is available for all CLI commands
data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
# 80% training, 20% testing to evaluate the model on unseen data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
# This is important for both SVM and KNN as they are sensitive to feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@train_app.command()
def svm(kernel: str = "linear", output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """
    Train a Support Vector Machine (SVM) classifier.
    
    This command is registered under the `train` subcommand, so users call:
      python script.py train svm --kernel rbf --output my_model.pkl
    
    Args:
        kernel: The SVM kernel type (default: "linear"). Options: "linear", "rbf", "poly", etc.
        output_file: Path where the trained SVM model will be saved (default: "model.ckpt")
    """
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@train_app.command()
def knn(n_neighbors: int = 5, output_file: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt") -> None:
    """
    Train a K-Nearest Neighbors (KNN) classifier.
    
    This command is registered under the `train` subcommand, so users call:
      python script.py train knn --n-neighbors 3 --output my_model.pkl
    
    Args:
        n_neighbors: The number of neighbors to use for KNN (default: 5)
        output_file: Path where the trained KNN model will be saved (default: "model.ckpt")
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    with open(output_file, "wb") as f:
        pickle.dump(model, f)


@app.command()
def evaluate(model_file: Annotated[str, typer.Argument()]):
    """
    Evaluate a trained model on the test set.
    
    This is a top-level command (not under a subcommand), so users call:
      python script.py evaluate my_model.pkl
    
    Args:
        model_file: Path to the trained model file to evaluate (required positional argument)
    """
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


if __name__ == "__main__":
    # Entry point for the CLI application
    # Typer routes commands based on the hierarchical structure:
    # - `python script.py train svm ...` routes to the svm() function in train_app
    # - `python script.py train knn ...` routes to the knn() function in train_app
    # - `python script.py evaluate ...` routes to the evaluate() function in app
    # - `python script.py --help` shows all available commands
    app()