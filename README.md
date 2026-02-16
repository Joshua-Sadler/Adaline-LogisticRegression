# Assignment 1 – Adaline and Logistic Regression
Name: Joshua Sadler

This project implements the following algorithms:

- Perceptron
- Adaline (Gradient Descent)
- Logistic Regression (Gradient Descent)
- Stochastic Gradient Descent (SGD)
- Mini-Batch SGD

No scikit-learn was used.

The project compares:
- Adaline vs Logistic Regression
- Iris vs Wine datasets
- GD vs SGD vs Mini-Batch SGD
- Binary vs Multiclass classification


# Textbook Reference & Original Code

This assignment is based on:

Machine Learning with PyTorch and Scikit-Learn  
by Sebastian Raschka

The original textbook example code can be found here:

https://github.com/rasbt/machine-learning-book

Specifically:
- Chapter 2 (Adaline & Perceptron)
- Chapter 3 (Logistic Regression)

# Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib

# Environment Setup

This project includes an `environment.yml` file that contains all required
dependencies and the correct Python version.

# How to Run the Programs

Always run commands from the project root directory.

Example:

`cd Adaline-LogisticRegression`


Task 1 & 2 – Binary Comparison (Iris & Wine)

`python -m experiments.comparison`

This will:
- Train Adaline and Logistic Regression
- Compare loss convergence
- Show performance plots


Task 3 – Multiclass Iris (Perceptron One-vs-Rest)

`python -m experiments.iris_multiclass_perceptron`

This will:
- Train 3 Perceptrons (One-vs-Rest)
- Plot decision boundaries
- Print multiclass accuracy


Task 4 – GD vs SGD vs Mini-Batch (Wine Dataset)

`python -m experiments.wine_gd_sgd_comparison`

This will:
- Train Logistic Regression using:
  - Full Gradient Descent
  - Stochastic Gradient Descent
  - Mini-Batch SGD
- Print timing results
- Plot loss convergence curves
