import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os

# Load preprocessed data
data_path = 'data/preprocessed_poultry_data.csv'
data = pd.read_csv(data_path)

# Reconstruct target column
disease_cols = [col for col in data.columns if col.startswith('disease_')]
data['target'] = 'Healthy'
for col in disease_cols:
    disease_name = col.replace('disease_', '')
    data.loc[data[col] == True, 'target'] = disease_name

# Features and target
X = data.drop(columns=disease_cols + ['target'])
y = data['target']

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Train on full training set (already done in grid_search)

# Evaluate on test set
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize feature importance
feature_importances = best_rf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.close()
print("Feature importance plot saved to feature_importance.png")

# Save model
os.makedirs('trained_models', exist_ok=True)
model_path = 'trained_models/poultry_model.pkl'
joblib.dump(best_rf, model_path)
print(f"Model saved to {model_path}")

# Summary
summary = {
    'best_params': grid_search.best_params_,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': conf_matrix.tolist(),
    'feature_importances': dict(zip(features, feature_importances))
}

print("Training and evaluation summary:")
print(summary)