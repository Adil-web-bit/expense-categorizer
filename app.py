# app.py
import streamlit as st
import joblib
import pandas as pd
from preprocess import clean_text, extract_amount_from_text

ADVANCED_MODEL_PATH = "models/complete_advanced_predictor.joblib"

class CompleteAdvancedPredictor:
    def __init__(self, path):
        try:
            self.models = joblib.load(path)
            print(f"✅ Loaded advanced models from {path}")
        except:
            print(f"❌ Failed to load {path}, falling back to simple model")
            self.models = joblib.load("models/expense_model.joblib")

@st.cache_resource
def load_advanced_model(path):
    return CompleteAdvancedPredictor(path)

st.title("🏷️ Advanced ML Expense Categorizer")
st.write("**Powered by XGBoost, Random Forest & Logistic Regression**")
st.write("**99.4% accuracy with multi-model predictions & confidence scores**")
st.write("Enter merchant and/or description; AI models will predict the expense category with confidence levels.")

# Display supported categories
st.sidebar.header("📊 Supported Categories")  
st.sidebar.write("""
- 🍕 **Food** - Restaurants, groceries, meals
- 🚗 **Transport** - Gas, Uber, parking, car expenses  
- 🛍️ **Shopping** - Retail, clothing, general purchases
- 🎬 **Entertainment** - Movies, streaming, games
- 💻 **Technology** - Electronics, software, gadgets
- ⚡ **Utilities** - Electric, water, internet bills
- 🏥 **Healthcare** - Medical, pharmacy, insurance
- 🏠 **Rent** - Housing, apartment, property costs
- ✈️ **Travel** - Hotels, flights, vacation expenses
- 🎓 **Education** - Tuition, courses, books
- 🛡️ **Insurance** - Health, car, life insurance
- 💪 **Fitness** - Gym, sports, workout gear
""")

st.markdown("---")

predictor = load_advanced_model(ADVANCED_MODEL_PATH)

# Use session state for examples
merchant_default = st.session_state.get('example_merchant', '')
description_default = st.session_state.get('example_description', '') 
amount_default = st.session_state.get('example_amount', '')

merchant = st.text_input("🏪 Merchant (optional)", value=merchant_default, help="e.g. McDonald's, Amazon, Shell")
description = st.text_input("📝 Description", value=description_default, help="e.g. 'Bought lunch', 'Monthly subscription', 'Gas fill-up'")
amount_text = st.text_input("💰 Amount (optional, leave blank to auto-extract)", value=amount_default, help="e.g. 25.99")

# Clear session state after using
for key in ['example_merchant', 'example_description', 'example_amount']:
    if key in st.session_state:
        del st.session_state[key]

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🔮 Predict Category", type="primary"):
        combined = (merchant + " " + description).strip()
        if not combined:
            st.warning("⚠️ Please enter merchant or description.")
        else:
            with st.spinner("🤖 Advanced AI models are analyzing your expense..."):
                text = clean_text(combined)
                # amount: prefer user input; else extract from description
                if amount_text.strip():
                    try:
                        amount = float(amount_text)
                    except:
                        amount = extract_amount_from_text(combined)
                else:
                    amount = extract_amount_from_text(combined)

                # Check if we have advanced models loaded
                if hasattr(predictor.models, 'keys'):
                    # Use advanced multi-model predictor
                    input_df = pd.DataFrame({"text": [text], "amount": [amount]})
                    
                    # Get predictions from all models
                    results = {}
                    for model_name, model in predictor.models.items():
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict(input_df)[0]
                            probs = model.predict_proba(input_df)[0]
                            max_prob = probs.max() * 100
                            results[model_name] = {'category': pred, 'confidence': max_prob}
                    
                    # Display multi-model results
                    st.success("🎯 **Advanced Multi-Model Predictions**")
                    
                    # Show results from each model
                    for model_name, result in results.items():
                        if result['confidence'] >= 90:
                            confidence_icon = "🎯"
                        elif result['confidence'] >= 70:
                            confidence_icon = "🤔"
                        else:
                            confidence_icon = "🤷"
                        
                        st.write(f"**{model_name}**: {result['category'].upper()} {confidence_icon} ({result['confidence']:.1f}%)")
                    
                    # Use best model (XGBoost if available, otherwise first model)
                    best_model_name = 'XGBoost' if 'XGBoost' in results else list(results.keys())[0]
                    best_result = results[best_model_name]
                    
                    st.info(f"🏆 **Best Prediction ({best_model_name})**: {best_result['category'].upper()} ({best_result['confidence']:.1f}%)")
                    
                    # Show top-3 predictions from best model
                    best_model = predictor.models[best_model_name]
                    all_probs = best_model.predict_proba(input_df)[0]
                    proba_df = pd.DataFrame({
                        "category": best_model.classes_, 
                        "probability": all_probs
                    }).sort_values("probability", ascending=False).head(3)
                    
                    proba_df["confidence"] = (proba_df["probability"] * 100).round(1).astype(str) + "%"
                    proba_df = proba_df[["category", "confidence"]].reset_index(drop=True)
                    
                    st.write("🏅 **Top-3 Predictions:**")
                    st.dataframe(proba_df, use_container_width=True, hide_index=True)
                    
                else:
                    # Fallback to simple model
                    input_df = pd.DataFrame({"text": [text], "amount": [amount]})
                    pred = predictor.models.predict(input_df)[0]
                    probs = predictor.models.predict_proba(input_df)[0]
                    proba_df = pd.DataFrame({"category": predictor.models.classes_, "probability": probs}).sort_values("probability", ascending=False)
                    
                    # Format probabilities as percentages
                    proba_df["confidence"] = (proba_df["probability"] * 100).round(1).astype(str) + "%"
                    proba_df = proba_df[["category", "confidence"]].reset_index(drop=True)

                    # Display results with nice formatting
                    st.success(f"🎯 **Predicted Category: {pred.upper()}**")
                    
                    # Show confidence level
                    top_confidence = probs[list(predictor.models.classes_).index(pred)] * 100
                    if top_confidence >= 90:
                        st.info(f"🎯 **High Confidence**: {top_confidence:.1f}%")
                    elif top_confidence >= 70:
                        st.info(f"🤔 **Medium Confidence**: {top_confidence:.1f}%")
                    else:
                        st.warning(f"🤷 **Low Confidence**: {top_confidence:.1f}%")
                    
                    st.write("📊 **All Category Predictions:**")
                    st.dataframe(proba_df, use_container_width=True, hide_index=True)
                
                # Show expense details
                st.write("---")
                st.write("📋 **Expense Details:**")
                st.write(f"- **Merchant**: {merchant if merchant else 'Not specified'}")
                st.write(f"- **Description**: {description if description else 'Not specified'}")
                st.write(f"- **Amount**: ${amount:.2f}")

with col2:
    if st.button("🔄 Clear", type="secondary"):
        st.rerun()

# Add some example inputs
st.markdown("---")
st.header("💡 Try These Examples")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🍕 Food Example"):
        st.session_state.example_merchant = "McDonald's"
        st.session_state.example_description = "Big Mac meal for lunch"
        st.session_state.example_amount = "12.99"

with example_col2:
    if st.button("🏥 Healthcare Example"):
        st.session_state.example_merchant = "CVS Pharmacy" 
        st.session_state.example_description = "prescription medication"
        st.session_state.example_amount = "45.50"

with example_col3:
    if st.button("✈️ Travel Example"):
        st.session_state.example_merchant = "Delta Airlines"
        st.session_state.example_description = "flight booking to New York"
        st.session_state.example_amount = "450.00"

# Auto-fill if example was clicked
if 'example_merchant' in st.session_state:
    st.rerun()
