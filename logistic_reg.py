import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the Digits dataset for demonstration
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train a simple Support Vector Machine (SVM) classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Save the trained SVM model
svm_model_path = '/home/sweta/mlops/models/m22aie204_svm_model.pkl'
joblib.dump(svm_model, svm_model_path)
print(f"SVM Model saved to {svm_model_path}")

# Define Logistic Regression solvers
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# Iterate over each solver and train Logistic Regression models
for solver in solvers:
    # Initialize Logistic Regression with the current solver
    lr = LogisticRegression(solver=solver, max_iter=1000)
    
    # Train the Logistic Regression model
    lr.fit(X_train, y_train)
    
    # Evaluate the model using cross-validation (5-fold CV)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    # Save the Logistic Regression model
    lr_model_filename = f'/app/models/m22aie204_lr_{solver}.joblib'
    joblib.dump(lr, lr_model_filename)
    print(f"Logistic Regression Model with solver {solver} saved to {lr_model_filename}")

    # Print CV results
    print(f'Solver: {solver}, Mean CV Score: {mean_cv_score}, Std CV Score: {std_cv_score}')