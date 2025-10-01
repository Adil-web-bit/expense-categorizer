# advanced_train.py - Enhanced ML models with transformers and ensemble methods
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Install required packages if not available
def install_dependencies():
    """Install required packages for advanced models"""
    try:
        import xgboost as xgb
        print("✓ XGBoost available")
    except ImportError:
        print("Installing XGBoost...")
        os.system("pip install xgboost")
    
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        print("✓ Transformers available")
    except ImportError:
        print("Installing transformers and torch...")
        os.system("pip install transformers torch")

# Advanced preprocessing with transformers
class AdvancedTextProcessor:
    def __init__(self, use_transformers=True):
        self.use_transformers = use_transformers
        self.bert_tokenizer = None
        self.bert_model = None
        
        if use_transformers:
            try:
                from transformers import DistilBertTokenizer, DistilBertModel
                import torch
                
                print("Loading DistilBERT model...")
                self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
                self.bert_model.eval()
                print("✓ DistilBERT loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load transformers: {e}")
                print("Falling back to TF-IDF")
                self.use_transformers = False
    
    def get_bert_embeddings(self, texts, max_length=128):
        """Get BERT embeddings for texts"""
        if not self.use_transformers:
            return None
            
        try:
            import torch
            embeddings = []
            
            for text in texts:
                # Tokenize and encode
                inputs = self.bert_tokenizer(
                    text, 
                    return_tensors='pt', 
                    max_length=max_length, 
                    truncation=True, 
                    padding='max_length'
                )
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    embeddings.append(embedding)
            
            return np.array(embeddings)
        except Exception as e:
            print(f"Error getting BERT embeddings: {e}")
            return None

# Enhanced model training with multiple algorithms
class AdvancedModelTrainer:
    def __init__(self, data_path="enhanced_expense_dataset.csv"):
        self.data_path = data_path
        self.models = {}
        self.model_scores = {}
        self.text_processor = AdvancedTextProcessor()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        from preprocess import prepare_dataframe
        
        print(f"Loading dataset: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Raw shape: {df.shape}")
        
        # Prepare dataframe
        df = prepare_dataframe(df, merchant_col='merchant', desc_col='description', amount_col='amount')
        df = df.dropna(subset=['category'])
        df = df[df['text'].str.len() > 0]
        
        print(f"After cleaning shape: {df.shape}")
        print(f"Categories: {sorted(df['category'].unique())}")
        
        return df
    
    def create_features(self, df):
        """Create advanced features for training"""
        print("Creating advanced features...")
        
        # Basic features
        X_text = df['text'].values
        X_amount = df[['amount']].values
        y = df['category'].values
        
        # TF-IDF features
        tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=15000,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        X_tfidf = tfidf.fit_transform(X_text)
        
        # BERT features (if available)
        X_bert = None
        if self.text_processor.use_transformers:
            print("Generating BERT embeddings...")
            X_bert = self.text_processor.get_bert_embeddings(X_text)
        
        # Combine features
        feature_dict = {
            'tfidf': X_tfidf,
            'amount': X_amount,
            'bert': X_bert,
            'text_processor': tfidf
        }
        
        return feature_dict, y
    
    def train_logistic_regression(self, X_tfidf, X_amount, y):
        """Train enhanced Logistic Regression"""
        print("\n=== Training Enhanced Logistic Regression ===")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('tfidf', 'passthrough', slice(0, X_tfidf.shape[1])),
                ('scaler', StandardScaler(), slice(X_tfidf.shape[1], X_tfidf.shape[1] + 1))
            ]
        )
        
        # Create pipeline with hyperparameter tuning
        pipeline = Pipeline([
            ('pre', preprocessor),
            ('clf', LogisticRegression(max_iter=3000, class_weight='balanced'))
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'clf__C': [0.1, 1.0, 10.0],
            'clf__solver': ['lbfgs', 'liblinear']
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_amount])
        
        grid_search.fit(X_combined, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_tfidf, X_amount, y):
        """Train Random Forest with hyperparameter tuning"""
        print("\n=== Training Advanced Random Forest ===")
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_amount]).toarray()  # RF needs dense arrays
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_combined, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, X_combined
    
    def train_xgboost(self, X_tfidf, X_amount, y):
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")
        
        try:
            import xgboost as xgb
            
            # Combine features
            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, X_amount]).toarray()
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_combined, y_encoded)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_, X_combined, y_encoded
            
        except ImportError:
            print("XGBoost not available, skipping...")
            return None, None, None
    
    def train_bert_classifier(self, X_bert, y):
        """Train classifier with BERT features"""
        if X_bert is None:
            print("BERT features not available, skipping...")
            return None
            
        print("\n=== Training BERT-based Classifier ===")
        
        # Simple classifier on BERT features
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        scores = cross_val_score(clf, X_bert, y, cv=5, scoring='accuracy')
        
        clf.fit(X_bert, y)
        print(f"BERT Classifier CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return clf
    
    def create_ensemble_model(self, models_dict):
        """Create ensemble voting classifier"""
        print("\n=== Creating Ensemble Model ===")
        
        available_models = [(name, model) for name, model in models_dict.items() if model is not None]
        
        if len(available_models) < 2:
            print("Not enough models for ensemble, skipping...")
            return None
        
        ensemble = VotingClassifier(
            estimators=available_models,
            voting='soft'  # Use probability voting
        )
        
        return ensemble
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities for top-k predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        self.model_scores[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return accuracy, y_pred, y_proba
    
    def save_advanced_model(self, best_model, feature_processor, model_name="advanced_expense_model"):
        """Save the best model with metadata"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(best_model, model_path)
        
        # Save feature processor
        processor_path = os.path.join(model_dir, f"{model_name}_processor.joblib")
        joblib.dump(feature_processor, processor_path)
        
        # Save metadata
        metadata = {
            'model_type': type(best_model).__name__,
            'timestamp': datetime.now().isoformat(),
            'accuracy': max(self.model_scores.values(), key=lambda x: x['accuracy'])['accuracy'],
            'features': 'TF-IDF + Amount + BERT (if available)',
            'categories': list(best_model.classes_) if hasattr(best_model, 'classes_') else None
        }
        
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAdvanced model saved to: {model_path}")
        print(f"Feature processor saved to: {processor_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path
    
    def train_all_models(self):
        """Train all available models and select the best one"""
        print("ADVANCED MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Install dependencies
        install_dependencies()
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Create features
        features, y = self.create_features(df)
        X_tfidf = features['tfidf']
        X_amount = features['amount']
        X_bert = features['bert']
        tfidf_processor = features['text_processor']
        
        # Split data
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_amount])
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train individual models
        models = {}
        
        # 1. Enhanced Logistic Regression
        try:
            lr_model = self.train_logistic_regression(X_tfidf[:-len(X_test)], X_amount[:-len(X_test)], y_train)
            models['logistic_regression'] = lr_model
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
        
        # 2. Random Forest
        try:
            rf_model, X_rf = self.train_random_forest(X_tfidf, X_amount, y)
            models['random_forest'] = rf_model
        except Exception as e:
            print(f"Error training Random Forest: {e}")
        
        # 3. XGBoost
        try:
            xgb_model, X_xgb, y_xgb = self.train_xgboost(X_tfidf, X_amount, y)
            if xgb_model:
                models['xgboost'] = xgb_model
        except Exception as e:
            print(f"Error training XGBoost: {e}")
        
        # 4. BERT Classifier
        if X_bert is not None:
            try:
                bert_model = self.train_bert_classifier(X_bert, y)
                if bert_model:
                    models['bert_classifier'] = bert_model
            except Exception as e:
                print(f"Error training BERT classifier: {e}")
        
        # Evaluate all models
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        for name, model in models.items():
            if model is None:
                continue
                
            try:
                # Prepare test data based on model type
                if name == 'bert_classifier' and X_bert is not None:
                    X_test_model = X_bert[-len(X_test):]
                else:
                    X_test_model = X_test
                
                accuracy, _, _ = self.evaluate_model(model, X_test_model, y_test, name)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Save best model
        if best_model:
            print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
            model_path = self.save_advanced_model(best_model, tfidf_processor, "advanced_expense_model")
            
            return best_model, best_model_name, best_accuracy
        else:
            print("No models were successfully trained!")
            return None, None, 0

# Enhanced prediction class with top-k predictions
class AdvancedPredictor:
    def __init__(self, model_path="models/advanced_expense_model.joblib"):
        self.model = joblib.load(model_path)
        self.processor = None
        
        # Try to load processor
        try:
            processor_path = model_path.replace('.joblib', '_processor.joblib')
            self.processor = joblib.load(processor_path)
        except:
            print("Warning: Could not load feature processor")
    
    def predict_with_probabilities(self, texts, amounts, top_k=3):
        """Make predictions with top-k probabilities"""
        # Process features
        if self.processor:
            X_text = self.processor.transform(texts)
        else:
            # Fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=10000)
            X_text = tfidf.fit_transform(texts)
        
        # Combine with amounts
        from scipy.sparse import hstack
        import numpy as np
        X_amount = np.array(amounts).reshape(-1, 1)
        X_combined = hstack([X_text, X_amount])
        
        # Make predictions
        predictions = self.model.predict(X_combined)
        probabilities = self.model.predict_proba(X_combined)
        
        # Get top-k predictions for each sample
        results = []
        for i in range(len(predictions)):
            # Get class probabilities
            probs = probabilities[i]
            classes = self.model.classes_
            
            # Sort by probability
            sorted_indices = np.argsort(probs)[::-1]
            
            top_predictions = []
            for j in range(min(top_k, len(classes))):
                idx = sorted_indices[j]
                top_predictions.append({
                    'category': classes[idx],
                    'probability': probs[idx],
                    'confidence': probs[idx] * 100
                })
            
            results.append({
                'top_prediction': predictions[i],
                'top_k_predictions': top_predictions,
                'all_probabilities': dict(zip(classes, probs))
            })
        
        return results

if __name__ == "__main__":
    # Train advanced models
    trainer = AdvancedModelTrainer()
    best_model, model_name, accuracy = trainer.train_all_models()
    
    if best_model:
        print(f"\n✅ Advanced model training completed successfully!")
        print(f"Best model: {model_name} with {accuracy:.1%} accuracy")
        
        # Test the advanced predictor
        print("\n🧪 Testing Advanced Predictor...")
        predictor = AdvancedPredictor()
        
        test_texts = ["McDonald's Big Mac meal", "Shell gas tank fill-up", "Apple Store iPhone purchase"]
        test_amounts = [12.99, 52.30, 999.00]
        
        results = predictor.predict_with_probabilities(test_texts, test_amounts, top_k=3)
        
        for i, result in enumerate(results):
            print(f"\nExample {i+1}: {test_texts[i]} (${test_amounts[i]})")
            print(f"Top prediction: {result['top_prediction']}")
            print("Top 3 predictions:")
            for pred in result['top_k_predictions']:
                print(f"  {pred['category']}: {pred['confidence']:.1f}%")
    else:
        print("❌ Model training failed!")