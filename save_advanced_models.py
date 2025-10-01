#!/usr/bin/env python3
"""
Save Advanced Models - Fixed Version
Creates and saves advanced ML models with proper pickling
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from advanced_predictor import AdvancedExpensePredictor

# Import preprocessing functions
from preprocess import prepare_dataframe

def save_advanced_models():
    print("🚀 TRAINING AND SAVING ADVANCED MODELS")
    print("=" * 50)
    
    # Load data
    print("📊 Loading enhanced dataset...")
    df = pd.read_csv('enhanced_expense_dataset.csv')
    df = prepare_dataframe(df, merchant_col='merchant', desc_col='description', amount_col='amount')
    df = df.dropna(subset=['category'])
    df = df[df['text'].str.len() > 0]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Categories: {len(df['category'].unique())}")
    
    # Enhanced TF-IDF features
    print("\n🔧 Creating enhanced TF-IDF features...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10000,
        min_df=2,
        stop_words='english'
    )
    
    X_text = tfidf.fit_transform(df['text'])
    X_amount = df[['amount']].values
    X = hstack([X_text, X_amount])
    y = df['category'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training features: {X_train.shape}")
    
    # Train XGBoost (best performing model)
    print("\n⚡ Training XGBoost...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train.toarray(), y_train_enc)
    xgb_pred_enc = xgb_model.predict(X_test.toarray())
    xgb_pred = le.inverse_transform(xgb_pred_enc)
    xgb_score = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_score:.3f} ({xgb_score*100:.1f}%)")
    
    # Create and save advanced predictor
    print(f"\n💾 Creating advanced predictor...")
    xgb_tuple = (xgb_model, le)
    advanced_predictor = AdvancedExpensePredictor(xgb_tuple, tfidf, 'xgboost')
    
    # Save models separately for better compatibility
    print(f"💾 Saving models...")
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(tfidf, 'models/advanced_tfidf_vectorizer.joblib')
    
    # Save a simple predictor function
    def create_predictor():
        """Factory function to create the advanced predictor"""
        saved_xgb = joblib.load('models/xgboost_model.joblib')
        saved_le = joblib.load('models/label_encoder.joblib')
        saved_tfidf = joblib.load('models/advanced_tfidf_vectorizer.joblib')
        return AdvancedExpensePredictor((saved_xgb, saved_le), saved_tfidf, 'xgboost')
    
    # Test the saved models
    print(f"\n🧪 Testing saved models...")
    test_predictor = create_predictor()
    
    test_cases = [
        ("McDonald's Big Mac meal", 12.99),
        ("Shell gas station", 45.20),
        ("Netflix monthly subscription", 15.99),
        ("Walmart grocery shopping", 127.85)
    ]
    
    for text, amount in test_cases:
        result = test_predictor.predict_with_probabilities(text, amount, top_k=3)
        print(f"\n'{text}' (${amount})")
        print(f"  Prediction: {result['prediction'].upper()} ({result['confidence']*100:.1f}%)")
        print(f"  Top 3:")
        for i, (cat, prob) in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {cat.upper()}: {prob*100:.1f}%")
    
    print(f"\n✅ ADVANCED MODELS SAVED SUCCESSFULLY!")
    print(f"   XGBoost Accuracy: {xgb_score:.1%}")
    print(f"   Enhanced TF-IDF Features: {X_text.shape[1]:,}")
    print(f"   Files Saved:")
    print(f"     - models/xgboost_model.joblib")
    print(f"     - models/label_encoder.joblib")
    print(f"     - models/advanced_tfidf_vectorizer.joblib")
    print(f"     - advanced_predictor.py (predictor class)")
    
    return test_predictor, xgb_score

if __name__ == "__main__":
    save_advanced_models()