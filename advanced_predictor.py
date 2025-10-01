#!/usr/bin/env python3
"""
Advanced Expense Predictor Class
For use with enhanced ML models and probability predictions
"""

import numpy as np
from scipy.sparse import hstack

class AdvancedExpensePredictor:
    """
    Advanced predictor that provides top-k predictions with probabilities
    Supports multiple model types: sklearn, random_forest, xgboost
    """
    
    def __init__(self, model, tfidf_vectorizer, model_type='sklearn'):
        self.model = model
        self.tfidf = tfidf_vectorizer
        self.model_type = model_type
        
        if model_type == 'xgboost':
            self.xgb_model, self.label_encoder = model
    
    def predict_with_probabilities(self, text, amount, top_k=3):
        """
        Predict expense category with probability scores
        
        Args:
            text (str): Description text to classify
            amount (float): Transaction amount
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Contains prediction, confidence, and top_k predictions
        """
        # Create feature vector
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
            # Sklearn models (Logistic Regression, Random Forest, etc.)
            if self.model_type == 'random_forest':
                combined_features = combined_features.toarray()
            
            prediction = self.model.predict(combined_features)[0]
            probabilities = self.model.predict_proba(combined_features)[0]
            prob_dict = dict(zip(self.model.classes_, [float(p) for p in probabilities]))
        
        # Get top k predictions
        top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return {
            'prediction': prediction,
            'confidence': float(prob_dict[prediction]),
            'top_predictions': [(cat, float(prob)) for cat, prob in top_predictions],
            'all_probabilities': prob_dict
        }
    
    def predict_simple(self, text, amount):
        """Simple prediction for backward compatibility"""
        result = self.predict_with_probabilities(text, amount, top_k=1)
        return result['prediction']