#!/usr/bin/env python3
"""
Quick Advanced Model Training
Implements enhanced ML models with probability predictions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Import preprocessing functions
from preprocess import prepare_dataframe

def train_advanced_models():
    print("🚀 ADVANCED MODEL TRAINING")
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
    
    # Train models
    models = {}
    scores = {}
    
    # 1. Enhanced Logistic Regression
    print("\n🤖 Training Enhanced Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_score = accuracy_score(y_test, lr_pred)
    models['logistic'] = lr
    scores['logistic'] = lr_score
    print(f"Logistic Regression: {lr_score:.3f}")
    
    # 2. Random Forest
    print("\n🌳 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train.toarray(), y_train)
    rf_pred = rf.predict(X_test.toarray())
    rf_score = accuracy_score(y_test, rf_pred)
    models['random_forest'] = rf
    scores['random_forest'] = rf_score
    print(f"Random Forest: {rf_score:.3f}")
    
    # 3. XGBoost
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
    models['xgboost'] = (xgb_model, le)
    scores['xgboost'] = xgb_score
    print(f"XGBoost: {xgb_score:.3f}")
    
    # Find best model
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    
    print(f"\n🏆 RESULTS:")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        status = "⭐ BEST" if name == best_name else ""
        print(f"  {name.upper():15s}: {score:.3f} ({score*100:.1f}%) {status}")
    
    # Create advanced predictor class
    class AdvancedPredictor:
        def __init__(self, model, tfidf_vectorizer, model_type='sklearn'):
            self.model = model
            self.tfidf = tfidf_vectorizer
            self.model_type = model_type
            if model_type == 'xgboost':
                self.xgb_model, self.label_encoder = model
            
        def predict_with_probabilities(self, text, amount, top_k=3):
            # Create features
            text_features = self.tfidf.transform([text])
            amount_features = np.array([[amount]])
            combined_features = hstack([text_features, amount_features])
            
            if self.model_type == 'xgboost':
                # XGBoost prediction
                combined_dense = combined_features.toarray()
                pred_enc = self.xgb_model.predict(combined_dense)[0]
                prediction = self.label_encoder.inverse_transform([pred_enc])[0]
                probabilities = self.xgb_model.predict_proba(combined_dense)[0]
                
                # Map back to original categories
                prob_dict = {}
                for i, prob in enumerate(probabilities):
                    category = self.label_encoder.inverse_transform([i])[0]
                    prob_dict[category] = float(prob)
            else:
                # Sklearn models
                if self.model_type == 'random_forest':
                    combined_features = combined_features.toarray()
                
                prediction = self.model.predict(combined_features)[0]
                probabilities = self.model.predict_proba(combined_features)[0]
                prob_dict = dict(zip(self.model.classes_, probabilities))
            
            # Get top k predictions
            top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            return {
                'prediction': prediction,
                'confidence': float(prob_dict[prediction]),
                'top_predictions': [(cat, float(prob)) for cat, prob in top_predictions],
                'all_probabilities': prob_dict
            }
    
    # Create and save advanced predictor
    best_model = models[best_name]
    model_type = 'xgboost' if best_name == 'xgboost' else ('random_forest' if best_name == 'random_forest' else 'sklearn')
    
    advanced_predictor = AdvancedPredictor(best_model, tfidf, model_type)
    
    # Save models
    print(f"\n💾 Saving advanced models...")
    joblib.dump(advanced_predictor, 'models/advanced_expense_predictor.joblib')
    joblib.dump(tfidf, 'models/advanced_tfidf_vectorizer.joblib')
    
    # Test predictions
    print(f"\n🧪 Testing advanced predictions:")
    test_cases = [
        ("McDonald's Big Mac meal", 12.99),
        ("Shell gas station", 45.20),
        ("Netflix monthly subscription", 15.99),
        ("Walmart grocery shopping", 127.85)
    ]
    
    for text, amount in test_cases:
        result = advanced_predictor.predict_with_probabilities(text, amount)
        print(f"\n'{text}' (${amount})")
        print(f"  Prediction: {result['prediction'].upper()} ({result['confidence']*100:.1f}%)")
        print(f"  Top 3:")
        for i, (cat, prob) in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {cat.upper()}: {prob*100:.1f}%")
    
    print(f"\n✅ ADVANCED TRAINING COMPLETE!")
    print(f"   Best Model: {best_name.upper()} ({best_score:.1%})")
    print(f"   Features: Enhanced TF-IDF + Amount")
    print(f"   Saved: models/advanced_expense_predictor.joblib")
    
    return advanced_predictor, best_name, best_score

if __name__ == "__main__":
    train_advanced_models()