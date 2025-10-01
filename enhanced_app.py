# enhanced_app.py - Advanced ML Expense Categorizer with Complete Features
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from preprocess import clean_text, extract_amount_from_text

# Configuration
ADVANCED_MODEL_PATH = "models/complete_advanced_predictor.joblib"

@st.cache_resource
def load_advanced_models():
    """Load all available models"""
    models = {}
    try:
        # Use the main pipeline model which includes preprocessing
        main_model = joblib.load("models/expense_model.joblib")
        models['Advanced Pipeline'] = main_model
        print("✅ Loaded advanced pipeline model with preprocessing")
        return models
    except Exception as e:
        print(f"❌ Failed to load pipeline model: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'expense_history' not in st.session_state:
        st.session_state.expense_history = []
    if 'example_merchant' not in st.session_state:
        st.session_state.example_merchant = ''
    if 'example_description' not in st.session_state:
        st.session_state.example_description = ''
    if 'example_amount' not in st.session_state:
        st.session_state.example_amount = ''

def predict_expense(models_dict, text, amount, merchant="", description=""):
    """Make prediction using advanced models"""
    input_df = pd.DataFrame({"text": [text], "amount": [amount]})
    
    if models_dict is None or len(models_dict) == 0:
        return {
            'prediction': 'Error - No Models Available',
            'confidence': 0.0,
            'top_3': pd.DataFrame(columns=['category', 'probability']),
            'all_models': {'No Models': {'category': 'Error', 'confidence': 0.0}},
            'best_model': 'None'
        }
    
    # Use advanced multi-model predictor
    results = {}
    for model_name, model in models_dict.items():
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict(input_df)[0]
                probs = model.predict_proba(input_df)[0]
                max_prob = probs.max() * 100
                results[model_name] = {'category': pred, 'confidence': max_prob}
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    if not results:
        return {
            'prediction': 'Prediction Error',
            'confidence': 0.0,
            'top_3': pd.DataFrame(columns=['category', 'probability']),
            'all_models': {'Error': {'category': 'Prediction Failed', 'confidence': 0.0}},
            'best_model': 'None'
        }
    
    # Use best model (XGBoost if available, otherwise first model)
    best_model_name = 'XGBoost' if 'XGBoost' in results else list(results.keys())[0]
    best_model = models_dict[best_model_name]
    
    # Get top-3 predictions
    all_probs = best_model.predict_proba(input_df)[0]
    proba_df = pd.DataFrame({
        "category": best_model.classes_, 
        "probability": all_probs
    }).sort_values("probability", ascending=False).head(3)
    
    top_category = proba_df.iloc[0]['category']
    top_confidence = proba_df.iloc[0]['probability'] * 100
    
    return {
        'prediction': top_category,
        'confidence': top_confidence,
        'top_3': proba_df,
        'all_models': results,
        'best_model': best_model_name
    }

def add_to_history(merchant, description, amount, prediction, confidence):
    """Add prediction to expense history"""
    timestamp = datetime.now()
    entry = {
        'timestamp': timestamp,
        'date': timestamp.strftime('%Y-%m-%d'),
        'merchant': merchant or 'Not specified',
        'description': description or 'Not specified',
        'amount': amount,
        'predicted_category': prediction,
        'confidence': confidence
    }
    st.session_state.expense_history.append(entry)

def process_bulk_upload(uploaded_file, models_dict):
    """Process bulk file upload for predictions"""
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel files."
        
        # Validate required columns
        required_cols = ['description']
        optional_cols = ['merchant', 'amount']
        
        if 'description' not in df.columns:
            return None, "File must contain 'description' column."
        
        # Fill missing optional columns
        if 'merchant' not in df.columns:
            df['merchant'] = ''
        if 'amount' not in df.columns:
            df['amount'] = 0.0
        
        # Process predictions
        predictions = []
        for _, row in df.iterrows():
            merchant = str(row.get('merchant', ''))
            description = str(row.get('description', ''))
            
            # Handle amount
            try:
                amount = float(row.get('amount', 0))
                if amount == 0:
                    amount = extract_amount_from_text(description)
            except:
                amount = extract_amount_from_text(description)
            
            # Clean and predict
            combined = (merchant + " " + description).strip()
            text = clean_text(combined)
            
            if text:  # Only predict if we have text
                result = predict_expense(models_dict, text, amount, merchant, description)
                predictions.append({
                    'merchant': merchant or 'Not specified',
                    'description': description or 'Not specified',
                    'amount': amount,
                    'predicted_category': result['prediction'],
                    'confidence': result['confidence']
                })
        
        return predictions, None
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def create_visualizations():
    """Create dashboard visualizations"""
    if not st.session_state.expense_history:
        st.warning("No expense history available for visualization.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.expense_history)
    
    # 1. Bar Chart: Spending by Category
    category_spending = df.groupby('predicted_category')['amount'].sum().reset_index()
    category_spending = category_spending.sort_values('amount', ascending=False)
    
    fig_bar = px.bar(
        category_spending, 
        x='predicted_category', 
        y='amount',
        title='💰 Total Spending by Category',
        labels={'predicted_category': 'Category', 'amount': 'Amount ($)'},
        color='amount',
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. Pie Chart: Percentage Distribution
    fig_pie = px.pie(
        category_spending, 
        values='amount', 
        names='predicted_category',
        title='📊 Expense Distribution by Categories'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 3. Line Chart: Daily Trends
    daily_spending = df.groupby('date')['amount'].sum().reset_index()
    daily_spending['date'] = pd.to_datetime(daily_spending['date'])
    daily_spending = daily_spending.sort_values('date')
    
    fig_line = px.line(
        daily_spending, 
        x='date', 
        y='amount',
        title='📈 Daily Spending Trends',
        labels={'date': 'Date', 'amount': 'Amount ($)'}
    )
    fig_line.update_traces(line_color='#ff6b6b', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 4. Summary Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Expenses", f"${df['amount'].sum():.2f}")
    with col2:
        st.metric("Total Transactions", len(df))
    with col3:
        st.metric("Average Amount", f"${df['amount'].mean():.2f}")
    with col4:
        st.metric("Categories Used", df['predicted_category'].nunique())

def create_download_buttons():
    """Create export functionality"""
    if not st.session_state.expense_history:
        st.warning("No expense history to export.")
        return
    
    df = pd.DataFrame(st.session_state.expense_history)
    
    # Prepare export DataFrame
    export_df = df[['date', 'merchant', 'description', 'amount', 'predicted_category', 'confidence']].copy()
    export_df.columns = ['Date', 'Merchant', 'Description', 'Amount', 'Category', 'Confidence (%)']
    export_df['Confidence (%)'] = export_df['Confidence (%)'].round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"expense_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Expense Predictions', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Expense Predictions']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(export_df.columns):
                max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
        
        excel_data = output.getvalue()
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"expense_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced ML Expense Categorizer",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    models_dict = load_advanced_models()
    
    # Main title
    st.title("🏷️ Advanced ML Expense Categorizer")
    st.markdown("**Powered by XGBoost, Random Forest & Logistic Regression • 99.4% Accuracy**")
    
    # Sidebar with categories and navigation
    with st.sidebar:
        st.header("📊 Supported Categories")  
        st.markdown("""
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
        st.subheader("📈 Statistics")
        if st.session_state.expense_history:
            total_expenses = sum(entry['amount'] for entry in st.session_state.expense_history)
            st.metric("Total Tracked", f"${total_expenses:.2f}")
            st.metric("Transactions", len(st.session_state.expense_history))
        else:
            st.info("No expenses tracked yet")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📊 Bulk Upload", "📋 History", "📈 Dashboard"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Expense Prediction")
        
        # Use session state for examples
        merchant_default = st.session_state.get('example_merchant', '')
        description_default = st.session_state.get('example_description', '') 
        amount_default = st.session_state.get('example_amount', '')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            merchant = st.text_input("🏪 Merchant (optional)", value=merchant_default, help="e.g. McDonald's, Amazon, Shell")
            description = st.text_input("📝 Description", value=description_default, help="e.g. 'Bought lunch', 'Monthly subscription', 'Gas fill-up'")
            amount_text = st.text_input("💰 Amount (optional, leave blank to auto-extract)", value=amount_default, help="e.g. 25.99")
        
        with col2:
            st.markdown("### 💡 Quick Examples")
            if st.button("🍕 Food Example"):
                st.session_state.example_merchant = "McDonald's"
                st.session_state.example_description = "Big Mac meal for lunch"
                st.session_state.example_amount = "12.99"
                st.rerun()
            
            if st.button("🏥 Healthcare Example"):
                st.session_state.example_merchant = "CVS Pharmacy" 
                st.session_state.example_description = "prescription medication"
                st.session_state.example_amount = "45.50"
                st.rerun()
            
            if st.button("✈️ Travel Example"):
                st.session_state.example_merchant = "Delta Airlines"
                st.session_state.example_description = "flight booking to New York"
                st.session_state.example_amount = "450.00"
                st.rerun()
        
        # Clear session state after using
        for key in ['example_merchant', 'example_description', 'example_amount']:
            if key in st.session_state:
                del st.session_state[key]
        
        col_predict, col_clear = st.columns([3, 1])
        
        with col_predict:
            if st.button("🔮 Predict Category", type="primary", use_container_width=True):
                combined = (merchant + " " + description).strip()
                if not combined:
                    st.warning("⚠️ Please enter merchant or description.")
                else:
                    with st.spinner("🤖 Advanced AI models are analyzing your expense..."):
                        text = clean_text(combined)
                        
                        # Handle amount
                        if amount_text.strip():
                            try:
                                amount = float(amount_text)
                            except:
                                amount = extract_amount_from_text(combined)
                        else:
                            amount = extract_amount_from_text(combined)
                        
                        # Make prediction
                        result = predict_expense(models_dict, text, amount, merchant, description)
                        
                        # Add to history
                        add_to_history(merchant, description, amount, result['prediction'], result['confidence'])
                        
                        # Display results
                        st.success(f"🎯 **Main Prediction: {result['prediction'].upper()}** ({result['confidence']:.1f}%)")
                        
                        # Show confidence level
                        if result['confidence'] >= 90:
                            st.info("🎯 **High Confidence Prediction**")
                        elif result['confidence'] >= 70:
                            st.info("🤔 **Medium Confidence Prediction**")
                        else:
                            st.warning("🤷 **Low Confidence Prediction**")
                        
                        # Multi-model comparison
                        st.subheader("🤖 Multi-Model Comparison")
                        if result['all_models'] and len(result['all_models']) > 0:
                            # Ensure we have at least 1 column and at most 4 columns for display
                            num_models = min(max(len(result['all_models']), 1), 4)
                            model_cols = st.columns(num_models)
                            for i, (model_name, model_result) in enumerate(result['all_models'].items()):
                                if i < num_models:  # Prevent index out of range
                                    with model_cols[i]:
                                        confidence_icon = "🎯" if model_result['confidence'] >= 90 else "🤔" if model_result['confidence'] >= 70 else "🤷"
                                        st.metric(
                                            f"{model_name} {confidence_icon}",
                                            model_result['category'].upper(),
                                            f"{model_result['confidence']:.1f}%"
                                        )
                        else:
                            st.warning("⚠️ No model predictions available")
                        
                        # Top-3 predictions
                        st.subheader("🏅 Top-3 Most Likely Categories")
                        if not result['top_3'].empty:
                            top3_df = result['top_3'].copy()
                            top3_df['confidence'] = (top3_df['probability'] * 100).round(1).astype(str) + "%"
                            top3_df = top3_df[['category', 'confidence']].reset_index(drop=True)
                            
                            # Create index labels based on actual number of rows
                            index_labels = ['🥇 1st', '🥈 2nd', '🥉 3rd']
                            top3_df.index = index_labels[:len(top3_df)]
                            st.dataframe(top3_df, use_container_width=True)
                        else:
                            st.warning("⚠️ No top predictions available")
                        
                        # Expense details
                        st.subheader("📋 Expense Details")
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        with detail_col1:
                            st.write(f"**Merchant**: {merchant or 'Not specified'}")
                        with detail_col2:
                            st.write(f"**Description**: {description or 'Not specified'}")
                        with detail_col3:
                            st.write(f"**Amount**: ${amount:.2f}")
        
        with col_clear:
            if st.button("🔄 Clear", type="secondary", use_container_width=True):
                st.rerun()
    
    # Tab 2: Bulk Upload
    with tab2:
        st.header("📊 Bulk Expense Prediction")
        st.write("Upload a CSV or Excel file with your expenses to get bulk predictions.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain 'description' column. Optional: 'merchant' and 'amount' columns."
        )
        
        if uploaded_file is not None:
            st.info(f"File uploaded: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
            
            if st.button("🚀 Process Bulk Predictions", type="primary"):
                with st.spinner("Processing bulk predictions..."):
                    predictions, error = process_bulk_upload(uploaded_file, models_dict)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success(f"✅ Successfully processed {len(predictions)} expenses!")
                        
                        # Add to session history
                        for pred in predictions:
                            add_to_history(
                                pred['merchant'],
                                pred['description'], 
                                pred['amount'],
                                pred['predicted_category'],
                                pred['confidence']
                            )
                        
                        # Show preview
                        st.subheader("📋 Prediction Results Preview")
                        preview_df = pd.DataFrame(predictions)
                        preview_df['confidence'] = preview_df['confidence'].round(1)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Show summary
                        category_summary = preview_df['predicted_category'].value_counts()
                        st.subheader("📊 Category Summary")
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.bar_chart(category_summary)
                        
                        with summary_col2:
                            for category, count in category_summary.items():
                                st.write(f"**{category}**: {count} expenses")
        
        # File format help
        with st.expander("📝 File Format Help"):
            st.write("""
            **Required Column:**
            - `description`: Text description of the expense
            
            **Optional Columns:**
            - `merchant`: Name of the merchant/store
            - `amount`: Expense amount (if not provided, will be extracted from description)
            
            **Example CSV format:**
            ```
            merchant,description,amount
            McDonald's,Big Mac meal,12.99
            Shell,Gas fill up,45.50
            Amazon,Office supplies purchase,89.99
            ```
            """)
    
    # Tab 3: History
    with tab3:
        st.header("📋 Expense History")
        
        if st.session_state.expense_history:
            # Display history table
            history_df = pd.DataFrame(st.session_state.expense_history)
            display_df = history_df[['date', 'merchant', 'description', 'amount', 'predicted_category', 'confidence']].copy()
            display_df.columns = ['Date', 'Merchant', 'Description', 'Amount ($)', 'Category', 'Confidence (%)']
            display_df['Confidence (%)'] = display_df['Confidence (%)'].round(1)
            display_df['Amount ($)'] = display_df['Amount ($)'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Quick Stats")
                total_amount = history_df['amount'].sum()
                avg_amount = history_df['amount'].mean()
                st.metric("Total Expenses", f"${total_amount:.2f}")
                st.metric("Average Amount", f"${avg_amount:.2f}")
                st.metric("Total Transactions", len(history_df))
            
            with col2:
                st.subheader("🏷️ Categories")
                category_counts = history_df['predicted_category'].value_counts()
                for category, count in category_counts.head(5).items():
                    percentage = (count / len(history_df)) * 100
                    st.write(f"**{category}**: {count} ({percentage:.1f}%)")
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.expense_history = []
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No expense history yet. Make some predictions to see them here!")
    
    # Tab 4: Dashboard
    with tab4:
        st.header("📈 Expense Dashboard")
        
        if st.session_state.expense_history:
            create_visualizations()
            
            st.markdown("---")
            st.subheader("💾 Export Data")
            create_download_buttons()
        else:
            st.info("No data available for dashboard. Start by making some expense predictions!")

if __name__ == "__main__":
    main()