#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:32:58 2024

@author: sirbizz
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load and prepare data
data = pd.read_csv("/Users/sirbizz/Desktop/Sku/car_data2.csv")
data = data.apply(lambda x: LabelEncoder().fit_transform(x.astype(str)) if x.dtype == type(object) else x)
data.fillna(data.median(), inplace=True)
data.drop('policy_id', axis=1, inplace=True, errors='ignore')

# Prepare features and target
X = data.drop('claim_status', axis=1)
y = LabelEncoder().fit_transform(data['claim_status']) if data['claim_status'].dtype == type(object) else data['claim_status']

# Handle class imbalance
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)

# Configure and train Random Forest with pruning parameters
param_grid = {
    'n_estimators': [100],

    'max_features': ['sqrt'],
    'max_depth': [10, 20, None],  # Adding different depths
    'min_samples_split': [2, 10, 20],  # Adding min_samples_split parameter
    'min_samples_leaf': [1, 5, 10],  # Adding min_samples_leaf parameter
    'criterion': ['gini']
}

# Enable parallelization by setting n_jobs=-1
best_rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid=param_grid, cv=5, n_jobs=-1).fit(X_train, y_train).best_estimator_

# Evaluate model
y_pred = best_rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate observed and expected counts
observed_counts = np.bincount(y_pred)
expected_counts = np.array([len(y_pred) / 2, len(y_pred) / 2])  # Assuming two classes and p=0.5 for each

# Construct contingency table
contingency_table = np.array([observed_counts, expected_counts])

# Perform Chi-squared test
chi2, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
print("Chi-squared Test Statistic:", chi2)
print("P-value:", p_value)
