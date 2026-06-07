import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

# Suppress ConvergenceWarnings for iterative training
warnings.filterwarnings("ignore")

def run_iris_machine():
    print("--- 1. Data Preparation ---")
    # Load Standard Iris dataset (3 species)
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # --- ADDING A 4TH SYNTHETIC SPECIES ---
    # We will create "Iris Hybrid" by modifying features of existing samples
    # to create a new, distinct category.
    print("Adding 4th species: 'Iris Hybrid'...")
    X_synthetic = X[:50] + np.random.normal(0, 0.5, size=(50, 4)) # Based on Setosa but shifted
    y_synthetic = np.full((50,), 3) # Label 3 for the new species
    
    X = np.vstack([X, X_synthetic])
    y = np.concatenate([y, y_synthetic])
    target_names.append("hybrid")
    
    # Save dataset to CSV as required
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    df.to_csv("iris_data.csv", index=False)
    print("Dataset (with 4 species) saved to iris_data.csv")

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples.")

    print("\n--- 2. Training (Weight Optimization) ---")
    # Using MLPClassifier to show iterative W adjustment
    # We will train it step by step to record the loss for visualization
    clf = MLPClassifier(
        hidden_layer_sizes=(10,), 
        max_iter=1, 
        warm_start=True, 
        random_state=42,
        learning_rate_init=0.01
    )
    
    losses = []
    iterations = 500
    for i in range(iterations):
        clf.fit(X_train, y_train)
        losses.append(clf.loss_)
        if i % 50 == 0:
            print(f"Iteration {i}: Loss = {clf.loss_:.4f}")

    # Plot Optimization (W adjustment / Loss reduction)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color='blue', linewidth=2)
    plt.title("Optimization Visual: Adjusting Weights (W) to Minimize Error", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss (Error)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig("optimization_w.png")
    print("Optimization plot saved as optimization_w.png")

    print("\n--- 3. Testing & Results ---")
    # Inject test data
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Final Machine Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=14)
    plt.xlabel("Predicted Species", fontsize=12)
    plt.ylabel("Actual Species", fontsize=12)
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

    return acc

if __name__ == "__main__":
    run_iris_machine()
