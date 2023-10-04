Decision Tree Classifier for Politician Party Prediction
Overview:
This repository contains a Decision Tree classifier designed to predict a politician's party based on their voting records. The model is built using a dataset that captures voting patterns, and the Decision Tree algorithm is employed to make predictions based on a series of decisions, much like navigating a flowchart.

Key Components:
DecisionTree Class: The core blueprint for creating the Decision Tree model.
Entropy Function: Measures the randomness or unpredictability of data labels.
Information Gain Function: Quantifies the reduction in uncertainty after making a decision.
Best Split Function: Determines the optimal decision at each step in the Decision Tree.
Evaluation Metrics: Tools to assess the model's performance, including accuracy, confusion matrix, precision, recall, and F1-score.
Learning Curve Plotting: Provides a visual representation of the model's learning progress.
Features:
Data Manipulation: Manages data loading, organization, and splitting into training and test sets.
Model Training: Uses the fit method to train the Decision Tree based on input data.
Prediction: The predict method allows for predictions on new data after the model is trained.
Performance Evaluation: Uses various metrics to gauge the model's accuracy and effectiveness.
Usage:
Clone the repository.
Ensure required libraries are installed.
Run the main program to train the Decision Tree and evaluate its performance on the test set.
Results:
The Decision Tree classifier has been evaluated over multiple runs, consistently achieving high accuracy rates. Precision, recall, and F1-score metrics further validate the model's robust performance across different classes. The learning curve visualization offers insights into the model's learning trajectory and its adaptability to the data.
