# train.py - Enhanced version for larger dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from preprocess import prepare_dataframe

# Use enhanced dataset by default, fallback to original if not found
ENHANCED_DATA_PATH = "enhanced_expense_dataset.csv"
ORIGINAL_DATA_PATH = "personal_expense_classification.csv"

if os.path.exists(ENHANCED_DATA_PATH):
    DATA_PATH = ENHANCED_DATA_PATH
    print(f"Using enhanced dataset: {ENHANCED_DATA_PATH}")
else:
    DATA_PATH = ORIGINAL_DATA_PATH
    print(f"Enhanced dataset not found, using original: {ORIGINAL_DATA_PATH}")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "expense_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# Basic clean + create 'text' and numeric 'amount'
df = prepare_dataframe(df, merchant_col='merchant', desc_col='description', amount_col='amount')

# drop rows without category
df = df.dropna(subset=['category'])
df = df[df['text'].str.len() > 0]  # remove empty text rows
print("After cleaning shape:", df.shape)

# X and y
X = df[['text', 'amount']]
y = df['category'].astype(str)

# train/test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Enhanced Preprocessor: Better TF-IDF for text + StandardScaler for amount
preprocessor = ColumnTransformer(transformers=[
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,3),           # Include trigrams for better context
        max_features=10000,          # More features for larger dataset
        min_df=2,                    # Ignore terms that appear in less than 2 documents
        max_df=0.95,                 # Ignore terms that appear in more than 95% of documents
        stop_words='english'         # Remove common English stop words
    ), 'text'),
    ('scaler', StandardScaler(), ['amount'])
], remainder='drop')

# Use Random Forest for better performance on larger dataset
# with ensemble methods typically performing better
pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=100,            # 100 trees for good performance
        max_depth=20,                # Prevent overfitting
        min_samples_split=5,         # Minimum samples to split
        min_samples_leaf=2,          # Minimum samples in leaf
        class_weight='balanced',      # Handle class imbalance
        random_state=42,
        n_jobs=-1                    # Use all available cores
    ))
])

# Alternative: Logistic Regression pipeline for comparison
logistic_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(
        max_iter=2000, 
        class_weight='balanced', 
        random_state=42,
        C=1.0
    ))
])

print("Training models and comparing performance...")

# Train Random Forest model
print("\n=== Training Random Forest Model ===")
pipeline.fit(X_train, y_train)
rf_pred = pipeline.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")

# Train Logistic Regression for comparison
print("\n=== Training Logistic Regression Model ===")
logistic_pipeline.fit(X_train, y_train)
lr_pred = logistic_pipeline.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")

# Choose the better performing model
if rf_accuracy >= lr_accuracy:
    best_pipeline = pipeline
    best_pred = rf_pred
    model_name = "Random Forest"
    print(f"\n=== Random Forest selected (accuracy: {rf_accuracy:.4f}) ===")
else:
    best_pipeline = logistic_pipeline
    best_pred = lr_pred
    model_name = "Logistic Regression"  
    print(f"\n=== Logistic Regression selected (accuracy: {lr_accuracy:.4f}) ===")

# Detailed evaluation of best model
print(f"\n=== {model_name} Model Evaluation ===")
print(f"Test Accuracy: {accuracy_score(y_test, best_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Show confusion matrix for categories with more readable format
unique_labels = sorted(best_pipeline.classes_)
cm = confusion_matrix(y_test, best_pred, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(cm_df)

# Show category-wise performance
print("\nCategory-wise Performance Summary:")
category_performance = pd.DataFrame(classification_report(y_test, best_pred, output_dict=True)).T
category_performance = category_performance.sort_values('support', ascending=False)
print(category_performance.round(3))

# Cross-validation score for robustness check
print("\n=== Cross-validation Assessment ===")
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance (if Random Forest is selected)
if hasattr(best_pipeline.named_steps['clf'], 'feature_importances_'):
    print("\n=== Feature Importance Analysis ===")
    
    # Get feature names from the preprocessor
    feature_names = []
    
    # TF-IDF features
    tfidf_features = best_pipeline.named_steps['pre'].named_transformers_['tfidf'].get_feature_names_out()
    feature_names.extend([f"text__{name}" for name in tfidf_features])
    
    # Amount feature  
    feature_names.append("amount")
    
    # Get feature importances
    importances = best_pipeline.named_steps['clf'].feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("Top 20 most important features:")
    print(importance_df.head(20))

# Save the best model
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\nSaved best model ({model_name}) to: {MODEL_PATH}")

# Save model performance metrics
metrics = {
    'model_type': model_name,
    'test_accuracy': float(accuracy_score(y_test, best_pred)),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'num_categories': len(unique_labels),
    'categories': unique_labels,
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nModel training completed successfully!")
print(f"Categories supported: {len(unique_labels)}")
print("Categories:", ", ".join(unique_labels))
