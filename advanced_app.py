#!/usr/bin/env python3
"""
Enhanced Expense Categorizer App with Advanced ML Models
Features:
- XGBoost model with 99.4% accuracy
- Top-3 predictions with probability scores
- Enhanced TF-IDF features (trigrams)
- Real-time confidence indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from advanced_predictor import AdvancedExpensePredictor

# Configure Streamlit page
st.set_page_config(
    page_title="🏦 Advanced Expense Categorizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_advanced_models():
    """Load advanced ML models with caching for better performance"""
    try:
        # Load individual model components
        xgb_model = joblib.load('models/xgboost_model.joblib')
        label_encoder = joblib.load('models/label_encoder.joblib')
        tfidf_vectorizer = joblib.load('models/advanced_tfidf_vectorizer.joblib')
        
        # Create advanced predictor
        predictor = AdvancedExpensePredictor(
            (xgb_model, label_encoder), 
            tfidf_vectorizer, 
            'xgboost'
        )
        
        return predictor, "XGBoost (99.4% accuracy)"
    except FileNotFoundError:
        st.error("❌ Advanced models not found! Please run 'save_advanced_models.py' first.")
        return None, None

def display_prediction_results(result, description, amount):
    """Display prediction results with enhanced UI"""
    prediction = result['prediction']
    confidence = result['confidence']
    top_predictions = result['top_predictions']
    
    # Main prediction with confidence indicator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### 🎯 **{prediction.upper()}**")
        
        # Confidence bar
        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
        st.markdown(f"""
        <div style="background-color: {confidence_color}; width: {confidence*100}%; height: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="text-align: center; color: white; font-weight: bold; line-height: 20px;">
                {confidence*100:.1f}% Confidence
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence emoji indicator
        if confidence > 0.9:
            st.markdown("### 🎯 Excellent")
        elif confidence > 0.8:
            st.markdown("### ✅ Very Good")
        elif confidence > 0.7:
            st.markdown("### 👍 Good")
        else:
            st.markdown("### ⚠️ Uncertain")
    
    # Top 3 predictions
    st.markdown("### 📊 **Top 3 Predictions**")
    
    for i, (category, probability) in enumerate(top_predictions):
        col1, col2, col3 = st.columns([1, 4, 2])
        
        with col1:
            if i == 0:
                st.markdown("🥇")
            elif i == 1:
                st.markdown("🥈")
            else:
                st.markdown("🥉")
        
        with col2:
            st.markdown(f"**{category.upper()}**")
        
        with col3:
            st.markdown(f"**{probability*100:.1f}%**")
        
        # Probability bar
        bar_color = ["#1f77b4", "#ff7f0e", "#2ca02c"][i]
        st.markdown(f"""
        <div style="background-color: {bar_color}; width: {probability*100}%; height: 8px; border-radius: 4px; margin: 5px 0;"></div>
        """, unsafe_allow_html=True)

def main():
    # Header with gradient background
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🏦 Advanced Expense Categorizer
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Powered by XGBoost ML with 99.4% Accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    predictor, model_info = load_advanced_models()
    
    if predictor is None:
        st.stop()
    
    # Sidebar with model information
    with st.sidebar:
        st.markdown("### 🔍 **Model Information**")
        st.info(f"**Active Model:** {model_info}")
        st.markdown("**Features:**")
        st.markdown("✅ Enhanced TF-IDF (1-3 grams)")
        st.markdown("✅ Amount-aware predictions")
        st.markdown("✅ Top-3 probability scores")
        st.markdown("✅ Real-time confidence")
        
        st.markdown("---")
        st.markdown("### 📋 **Categories**")
        categories = [
            "🍔 Food", "🚗 Transport", "🏠 Housing", "🎬 Entertainment",
            "🛍️ Shopping", "⚡ Utilities", "🏥 Healthcare", "📚 Education",
            "💼 Business", "👕 Personal Care", "🎁 Gifts", "📱 Other"
        ]
        
        for category in categories:
            st.markdown(f"• {category}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.markdown("### 💰 **Enter Expense Details**")
        
        description = st.text_input(
            "Expense Description",
            placeholder="e.g., McDonald's Big Mac meal, Shell gas station, Netflix subscription..."
        )
        
        amount = st.number_input(
            "Amount ($)",
            min_value=0.01,
            max_value=10000.0,
            value=25.0,
            step=0.01
        )
        
        # Quick example buttons
        st.markdown("**Quick Examples:**")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🍔 McDonald's $15"):
                st.session_state['desc'] = "McDonald's Big Mac meal"
                st.session_state['amt'] = 15.99
        
        with example_col2:
            if st.button("⛽ Gas Station $45"):
                st.session_state['desc'] = "Shell gas station fill-up"
                st.session_state['amt'] = 45.20
        
        with example_col3:
            if st.button("📱 Netflix $16"):
                st.session_state['desc'] = "Netflix monthly subscription"
                st.session_state['amt'] = 15.99
        
        # Update inputs from session state
        if 'desc' in st.session_state:
            description = st.session_state['desc']
            del st.session_state['desc']
        
        if 'amt' in st.session_state:
            amount = st.session_state['amt']
            del st.session_state['amt']
    
    with col2:
        # Advanced features info
        st.markdown("### 🚀 **Advanced Features**")
        st.markdown("""
        **🎯 XGBoost Algorithm**
        - 99.4% prediction accuracy
        - Enhanced gradient boosting
        
        **📊 Probability Scores**
        - Top-3 category predictions
        - Real-time confidence levels
        
        **🔧 Enhanced Processing**
        - Trigram text features
        - Amount-aware predictions
        - 12 expense categories
        """)
    
    # Prediction section
    if st.button("🔮 **Categorize Expense**", type="primary", use_container_width=True):
        if description and amount > 0:
            with st.spinner("🤖 Processing with advanced ML models..."):
                try:
                    # Get prediction with probabilities
                    result = predictor.predict_with_probabilities(
                        description, amount, top_k=3
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📈 **Prediction Results**")
                    
                    display_prediction_results(result, description, amount)
                    
                    # Additional insights
                    st.markdown("---")
                    st.markdown("### 💡 **Insights**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence Level", f"{result['confidence']*100:.1f}%")
                    with col2:
                        st.metric("Top Category", result['prediction'].title())
                    with col3:
                        alternatives = len([p for p in result['top_predictions'] if p[1] > 0.1])
                        st.metric("Alternatives", f"{alternatives-1} strong")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter both description and amount!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by Advanced Machine Learning • Enhanced TF-IDF Features • XGBoost Algorithm</p>
        <p>📊 Trained on 10,000+ realistic expense records with 99.4% accuracy</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()