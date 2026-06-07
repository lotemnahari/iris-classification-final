# Product Requirements Document (PRD): Iris Classification Machine

## 1. Overview
The goal of this project is to develop a high-performance classification system capable of identifying species of the Iris flower based on botanical features. The system must demonstrate the learning process, evaluate its performance on unseen data, and present results in a professional, "marketable" format.

## 2. Target Audience
This project is prepared for academic review (Professor) and as a showcase of machine learning capabilities.

## 3. Functional Requirements

### 3.1 Data Management
- **Dataset:** The standard Iris flower dataset (Sepal Length, Sepal Width, Petal Length, Petal Width).
- **Split:** Implementation of a strict **80% training** and **20% testing** partition.

### 3.2 Classification Machine
- **Algorithm:** Use a Multi-Layer Perceptron (MLP) or similar iterative optimization model that allows for weight ($W$) adjustment visualization.
- **Task:** Multi-class classification into three species: *Setosa*, *Versicolor*, and *Virginica*.

### 3.3 Visualizations & Outputs
- **Confusion Matrix:** A heatmapped visualization of actual vs. predicted classes for the test set.
- **Optimization Plot:** A graph showing the convergence of the model (how weights $W$ adjust or how loss decreases) over iterations.
- **Accuracy Metric:** Clear display of the final classification accuracy.

## 4. Deliverables
- `iris_classifier.py`: The core machine code.
- `iris_data.csv`: The dataset used for training and testing.
- `PRD_IRIS.md`: This document.
- `IRIS_SUMMARY_REPORT.md`: A comprehensive, marketing-oriented report.
- `confusion_matrix.png`: Visual output of the test phase.
- `optimization_w.png`: Visual output of the learning process.

## 5. Technical Stack
- **Language:** Python
- **Core Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## 6. Success Criteria
- [ ] Model achieves >90% accuracy on the test set.
- [ ] All visualizations are generated and saved as PNGs.
- [ ] The summary report is complete and "marketable."
