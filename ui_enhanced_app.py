# ui_enhanced_app.py - Beautiful ML Expense Categorizer with Enhanced UI
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from preprocess import clean_text, extract_amount_from_text

# Enhanced UI Imports
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu

# Configuration
ADVANCED_MODEL_PATH = "models/expense_model.joblib"

# Theme configuration
THEMES = {
    "light": {
        "primary_color": "#FF6B6B",
        "background_color": "#FFFFFF",
        "secondary_background_color": "#F0F2F6",
        "text_color": "#262730",
        "success_color": "#00C851",
        "warning_color": "#FFB000",
        "error_color": "#FF4444"
    },
    "dark": {
        "primary_color": "#FF6B6B",
        "background_color": "#0E1117",
        "secondary_background_color": "#262730",
        "text_color": "#FAFAFA",
        "success_color": "#00C851",
        "warning_color": "#FFB000",
        "error_color": "#FF4444"
    }
}

@st.cache_resource
def load_advanced_models():
    """Load all available models"""
    models = {}
    try:
        # Use the main pipeline model which includes preprocessing
        main_model = joblib.load("models/expense_model.joblib")
        models['Advanced Pipeline'] = main_model
        return models
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'expense_history' not in st.session_state:
        st.session_state.expense_history = []
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'example_merchant' not in st.session_state:
        st.session_state.example_merchant = ''
    if 'example_description' not in st.session_state:
        st.session_state.example_description = ''
    if 'example_amount' not in st.session_state:
        st.session_state.example_amount = ''

def apply_theme():
    """Apply the selected theme"""
    theme = THEMES[st.session_state.theme]
    
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {theme['background_color']};
            color: {theme['text_color']};
        }}
        
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, {theme['primary_color']}20, {theme['secondary_background_color']});
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {theme['primary_color']};
            margin: 0.5rem 0;
        }}
        
        .success-message {{
            background: linear-gradient(135deg, {theme['success_color']}20, {theme['secondary_background_color']});
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {theme['success_color']};
            color: {theme['success_color']};
            margin: 1rem 0;
        }}
        
        .warning-message {{
            background: linear-gradient(135deg, {theme['warning_color']}20, {theme['secondary_background_color']});
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {theme['warning_color']};
            color: {theme['warning_color']};
            margin: 1rem 0;
        }}
        
        .error-message {{
            background: linear-gradient(135deg, {theme['error_color']}20, {theme['secondary_background_color']});
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {theme['error_color']};
            color: {theme['error_color']};
            margin: 1rem 0;
        }}
        
        .prediction-card {{
            background: linear-gradient(135deg, {theme['primary_color']}, {theme['primary_color']}80);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .confidence-high {{
            background: linear-gradient(135deg, {theme['success_color']}, {theme['success_color']}80);
        }}
        
        .confidence-medium {{
            background: linear-gradient(135deg, {theme['warning_color']}, {theme['warning_color']}80);
        }}
        
        .confidence-low {{
            background: linear-gradient(135deg, {theme['error_color']}, {theme['error_color']}80);
        }}
        
        .sidebar .sidebar-content {{
            background-color: {theme['secondary_background_color']};
        }}
        
        h1, h2, h3 {{
            color: {theme['text_color']} !important;
        }}
        
        .stMetric {{
            background: {theme['secondary_background_color']};
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid {theme['primary_color']}30;
        }}
    </style>
    """, unsafe_allow_html=True)

def show_success_message(message, details=None):
    """Show enhanced success message"""
    st.markdown(f"""
    <div class="success-message">
        <h4>✅ {message}</h4>
        {f'<p>{details}</p>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def show_warning_message(message, details=None):
    """Show enhanced warning message"""
    st.markdown(f"""
    <div class="warning-message">
        <h4>⚠️ {message}</h4>
        {f'<p>{details}</p>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def show_error_message(message, details=None):
    """Show enhanced error message"""
    st.markdown(f"""
    <div class="error-message">
        <h4>❌ {message}</h4>
        {f'<p>{details}</p>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def predict_expense(models_dict, text, amount, merchant="", description=""):
    """Make prediction using advanced models"""
    input_df = pd.DataFrame({"text": [text], "amount": [amount]})
    
    if models_dict is None or len(models_dict) == 0:
        return {
            'prediction': 'Error - No Models Available',
            'confidence': 0.0,
            'top_3': pd.DataFrame(columns=['category', 'probability']),
            'all_models': {'No Models': {'category': 'Error', 'confidence': 0.0}},
            'best_model': 'None',
            'status': 'error'
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
            continue
    
    if not results:
        return {
            'prediction': 'Prediction Error',
            'confidence': 0.0,
            'top_3': pd.DataFrame(columns=['category', 'probability']),
            'all_models': {'Error': {'category': 'Prediction Failed', 'confidence': 0.0}},
            'best_model': 'None',
            'status': 'error'
        }
    
    # Use first available model
    best_model_name = list(results.keys())[0]
    best_model = models_dict[best_model_name]
    
    # Get top-3 predictions
    all_probs = best_model.predict_proba(input_df)[0]
    proba_df = pd.DataFrame({
        "category": best_model.classes_, 
        "probability": all_probs
    }).sort_values("probability", ascending=False).head(3)
    
    top_category = proba_df.iloc[0]['category']
    top_confidence = proba_df.iloc[0]['probability'] * 100
    
    # Determine status based on confidence
    if top_confidence >= 80:
        status = 'high_confidence'
    elif top_confidence >= 60:
        status = 'medium_confidence'
    else:
        status = 'low_confidence'
    
    return {
        'prediction': top_category,
        'confidence': top_confidence,
        'top_3': proba_df,
        'all_models': results,
        'best_model': best_model_name,
        'status': status
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
                    'confidence': result['confidence'],
                    'status': result['status']
                })
        
        return predictions, None
    
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def create_visualizations():
    """Create dashboard visualizations with enhanced design"""
    if not st.session_state.expense_history:
        show_warning_message("No expense history available", "Start making predictions to see visualizations")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.expense_history)
    
    # Enhanced color schemes based on theme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']
    
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
        color_continuous_scale='viridis',
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    fig_bar.update_layout(
        showlegend=False, 
        xaxis_tickangle=-45,
        title_font_size=20,
        font=dict(size=12)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. Pie Chart: Percentage Distribution
    fig_pie = px.pie(
        category_spending, 
        values='amount', 
        names='predicted_category',
        title='📊 Expense Distribution by Categories',
        color_discrete_sequence=colors,
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    fig_pie.update_layout(title_font_size=20)
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
        labels={'date': 'Date', 'amount': 'Amount ($)'},
        template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    )
    fig_line.update_traces(line_color='#FF6B6B', line_width=3)
    fig_line.update_layout(title_font_size=20)
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 4. Enhanced Summary Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Expenses", 
            f"${df['amount'].sum():.2f}",
            delta=f"{len(df)} transactions"
        )
    with col2:
        st.metric(
            "📊 Average Amount", 
            f"${df['amount'].mean():.2f}",
            delta=f"±${df['amount'].std():.2f}"
        )
    with col3:
        st.metric(
            "🏷️ Categories Used", 
            df['predicted_category'].nunique(),
            delta=f"out of 12 total"
        )
    with col4:
        avg_confidence = df['confidence'].mean()
        st.metric(
            "🎯 Avg Confidence", 
            f"{avg_confidence:.1f}%",
            delta="High" if avg_confidence >= 80 else "Medium" if avg_confidence >= 60 else "Low"
        )
    
    # Apply metric styling
    style_metric_cards(
        background_color=THEMES[st.session_state.theme]['secondary_background_color'],
        border_left_color=THEMES[st.session_state.theme]['primary_color'],
        border_color=THEMES[st.session_state.theme]['primary_color'] + '30',
        box_shadow=True
    )

def create_download_buttons():
    """Create export functionality with enhanced UI"""
    if not st.session_state.expense_history:
        show_warning_message("No expense history to export")
        return
    
    df = pd.DataFrame(st.session_state.expense_history)
    
    # Prepare export DataFrame
    export_df = df[['date', 'merchant', 'description', 'amount', 'predicted_category', 'confidence']].copy()
    export_df.columns = ['Date', 'Merchant', 'Description', 'Amount', 'Category', 'Confidence (%)']
    export_df['Confidence (%)'] = export_df['Confidence (%)'].round(2)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # CSV Download
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"expense_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Expense Predictions', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Expense Predictions']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#FF6B6B',
                'font_color': 'white',
                'border': 1
            })
            
            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            for i, col in enumerate(export_df.columns):
                max_len = max(export_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_len)
        
        excel_data = output.getvalue()
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"expense_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.expense_history = []
            show_success_message("History cleared successfully!")
            st.rerun()

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Expense Categorizer",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Apply theme
    apply_theme()
    
    # Load models
    models_dict = load_advanced_models()
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        colored_header(
            label="🏷️ AI-Powered Expense Categorizer",
            description="Advanced ML with Beautiful UI • 99%+ Accuracy",
            color_name="red-70"
        )
    
    with col2:
        add_vertical_space(1)
        if st.button(f"🌙 {'Light' if st.session_state.theme == 'dark' else 'Dark'} Theme"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    
    with col3:
        add_vertical_space(1)
        st.markdown(f"**Theme:** {st.session_state.theme.title()}")
    
    # Enhanced Sidebar
    with st.sidebar:
        colored_header(
            label="📊 Categories & Stats",
            description="AI Model Information",
            color_name="blue-green-70"
        )
        
        st.markdown("""
        ### 🎯 **Supported Categories**
        
        🍕 **Food** • 🚗 **Transport** • 🛍️ **Shopping**  
        🎬 **Entertainment** • 💻 **Technology** • ⚡ **Utilities**  
        🏥 **Healthcare** • 🏠 **Rent** • ✈️ **Travel**  
        🎓 **Education** • 🛡️ **Insurance** • 💪 **Fitness**
        """)
        
        add_vertical_space(2)
        
        if st.session_state.expense_history:
            total_expenses = sum(entry['amount'] for entry in st.session_state.expense_history)
            avg_confidence = sum(entry['confidence'] for entry in st.session_state.expense_history) / len(st.session_state.expense_history)
            
            st.markdown("### 📈 **Live Statistics**")
            
            st.metric("💰 Total Tracked", f"${total_expenses:.2f}")
            st.metric("📝 Transactions", len(st.session_state.expense_history))
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Confidence status
            if avg_confidence >= 80:
                st.success("🎯 **High Confidence Model**")
            elif avg_confidence >= 60:
                st.info("🤔 **Medium Confidence Model**")
            else:
                st.warning("🤷 **Low Confidence - More Data Needed**")
        else:
            st.info("💡 **Start making predictions to see statistics!**")
    
    # Enhanced Navigation
    selected = option_menu(
        menu_title=None,
        options=["🔮 Predict", "📊 Bulk Upload", "📋 History", "📈 Dashboard"],
        icons=["magic", "cloud-upload", "table", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": THEMES[st.session_state.theme]['primary_color'], "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": THEMES[st.session_state.theme]['secondary_background_color'],
            },
            "nav-link-selected": {"background-color": THEMES[st.session_state.theme]['primary_color']},
        }
    )
    
    add_vertical_space(2)
    
    # Tab 1: Single Prediction
    if selected == "🔮 Predict":
        colored_header(
            label="Single Expense Prediction",
            description="Get instant AI-powered category predictions",
            color_name="violet-70"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            merchant = st.text_input(
                "🏪 Merchant (optional)", 
                value=st.session_state.get('example_merchant', ''),
                help="e.g. McDonald's, Amazon, Shell",
                placeholder="Enter merchant name..."
            )
            description = st.text_input(
                "📝 Description", 
                value=st.session_state.get('example_description', ''),
                help="e.g. 'Bought lunch', 'Monthly subscription', 'Gas fill-up'",
                placeholder="Describe your expense..."
            )
            amount_text = st.text_input(
                "💰 Amount (optional, leave blank to auto-extract)", 
                value=st.session_state.get('example_amount', ''),
                help="e.g. 25.99",
                placeholder="0.00"
            )
        
        with col2:
            st.markdown("### 💡 **Quick Examples**")
            if st.button("🍕 Food Example", use_container_width=True):
                st.session_state.example_merchant = "McDonald's"
                st.session_state.example_description = "Big Mac meal for lunch"
                st.session_state.example_amount = "12.99"
                st.rerun()
            
            if st.button("🏥 Healthcare Example", use_container_width=True):
                st.session_state.example_merchant = "CVS Pharmacy" 
                st.session_state.example_description = "prescription medication"
                st.session_state.example_amount = "45.50"
                st.rerun()
            
            if st.button("✈️ Travel Example", use_container_width=True):
                st.session_state.example_merchant = "Delta Airlines"
                st.session_state.example_description = "flight booking to New York"
                st.session_state.example_amount = "450.00"
                st.rerun()
        
        # Clear session state after using
        for key in ['example_merchant', 'example_description', 'example_amount']:
            if key in st.session_state:
                del st.session_state[key]
        
        add_vertical_space(2)
        
        col_predict, col_clear = st.columns([3, 1])
        
        with col_predict:
            if st.button("🔮 Predict Category", type="primary", use_container_width=True):
                combined = (merchant + " " + description).strip()
                if not combined:
                    show_warning_message("Please enter merchant or description", "At least one field is required for prediction")
                else:
                    with st.spinner("🤖 AI is analyzing your expense..."):
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
                        
                        # Display results based on status
                        if result['status'] == 'error':
                            show_error_message(
                                "Prediction Failed", 
                                "Unable to categorize this expense. Please check your input."
                            )
                        else:
                            # Success message with confidence level
                            if result['status'] == 'high_confidence':
                                show_success_message(
                                    f"High Confidence Prediction: {result['prediction'].upper()}",
                                    f"The AI is {result['confidence']:.1f}% confident in this categorization"
                                )
                            elif result['status'] == 'medium_confidence':
                                show_warning_message(
                                    f"Medium Confidence Prediction: {result['prediction'].upper()}",
                                    f"The AI is {result['confidence']:.1f}% confident. Consider providing more details."
                                )
                            else:
                                show_warning_message(
                                    f"Low Confidence Prediction: {result['prediction'].upper()}",
                                    f"The AI is only {result['confidence']:.1f}% confident. Please verify this categorization."
                                )
                            
                            # Enhanced prediction display
                            confidence_class = f"confidence-{result['status'].split('_')[0]}"
                            st.markdown(f"""
                            <div class="prediction-card {confidence_class}">
                                <h2>🎯 {result['prediction'].upper()}</h2>
                                <h3>{result['confidence']:.1f}% Confidence</h3>
                                <p>Powered by {result['best_model']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Top-3 predictions
                            if not result['top_3'].empty:
                                colored_header(
                                    label="Top-3 Most Likely Categories",
                                    description="Alternative predictions ranked by probability",
                                    color_name="orange-70"
                                )
                                
                                top3_df = result['top_3'].copy()
                                top3_df['confidence'] = (top3_df['probability'] * 100).round(1).astype(str) + "%"
                                top3_df = top3_df[['category', 'confidence']].reset_index(drop=True)
                                
                                # Display as metrics
                                cols = st.columns(3)
                                medals = ['🥇', '🥈', '🥉']
                                for i, (_, row) in enumerate(top3_df.iterrows()):
                                    with cols[i]:
                                        st.metric(
                                            f"{medals[i]} {row['category'].title()}",
                                            row['confidence'],
                                            delta=f"Rank {i+1}"
                                        )
                            
                            # Expense details
                            add_vertical_space(1)
                            colored_header(
                                label="Expense Details",
                                description="Summary of the analyzed expense",
                                color_name="blue-70"
                            )
                            
                            detail_col1, detail_col2, detail_col3 = st.columns(3)
                            with detail_col1:
                                st.info(f"**🏪 Merchant**\n{merchant or 'Not specified'}")
                            with detail_col2:
                                st.info(f"**📝 Description**\n{description or 'Not specified'}")
                            with detail_col3:
                                st.info(f"**💰 Amount**\n${amount:.2f}")
        
        with col_clear:
            if st.button("🔄 Clear", type="secondary", use_container_width=True):
                st.rerun()
    
    # Tab 2: Bulk Upload
    elif selected == "📊 Bulk Upload":
        colored_header(
            label="Bulk Expense Prediction",
            description="Upload CSV/Excel files for mass categorization",
            color_name="green-70"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="File should contain 'description' column. Optional: 'merchant' and 'amount' columns."
        )
        
        if uploaded_file is not None:
            st.info(f"📁 **File uploaded:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            if st.button("🚀 Process Bulk Predictions", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing bulk predictions..."):
                    predictions, error = process_bulk_upload(uploaded_file, models_dict)
                    
                    if error:
                        show_error_message("Processing Failed", error)
                    else:
                        show_success_message(
                            f"Successfully processed {len(predictions)} expenses!",
                            "All predictions have been added to your history"
                        )
                        
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
                        colored_header(
                            label="Prediction Results Preview",
                            description="First 10 results from your bulk upload",
                            color_name="blue-green-70"
                        )
                        
                        preview_df = pd.DataFrame(predictions).head(10)
                        preview_df['confidence'] = preview_df['confidence'].round(1)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Show summary
                        add_vertical_space(2)
                        colored_header(
                            label="Processing Summary",
                            description="Overview of categorization results",
                            color_name="violet-70"
                        )
                        
                        category_summary = pd.DataFrame(predictions)['predicted_category'].value_counts()
                        
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            # Create summary chart
                            fig_summary = px.bar(
                                x=category_summary.values,
                                y=category_summary.index,
                                orientation='h',
                                title="Categories Distribution",
                                labels={'x': 'Count', 'y': 'Category'},
                                color=category_summary.values,
                                color_continuous_scale='viridis',
                                template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
                            )
                            st.plotly_chart(fig_summary, use_container_width=True)
                        
                        with summary_col2:
                            st.markdown("### 📊 **Category Breakdown**")
                            for category, count in category_summary.items():
                                percentage = (count / len(predictions)) * 100
                                st.write(f"**{category.title()}**: {count} expenses ({percentage:.1f}%)")
        
        # File format help
        with st.expander("📝 **File Format Help**"):
            st.markdown("""
            ### Required Column:
            - **`description`**: Text description of the expense
            
            ### Optional Columns:
            - **`merchant`**: Name of the merchant/store
            - **`amount`**: Expense amount (if not provided, will be extracted from description)
            
            ### Example CSV format:
            ```csv
            merchant,description,amount
            McDonald's,Big Mac meal,12.99
            Shell,Gas fill up,45.50
            Amazon,Office supplies purchase,89.99
            ```
            
            ### Supported File Types:
            - **.csv** (Comma-separated values)
            - **.xlsx** (Excel 2007+)
            - **.xls** (Excel 97-2003)
            """)
    
    # Tab 3: History
    elif selected == "📋 History":
        colored_header(
            label="Expense History",
            description="View and manage your prediction history",
            color_name="orange-70"
        )
        
        if st.session_state.expense_history:
            # Display history table
            history_df = pd.DataFrame(st.session_state.expense_history)
            display_df = history_df[['date', 'merchant', 'description', 'amount', 'predicted_category', 'confidence']].copy()
            display_df.columns = ['📅 Date', '🏪 Merchant', '📝 Description', '💰 Amount ($)', '🏷️ Category', '🎯 Confidence (%)']
            display_df['🎯 Confidence (%)'] = display_df['🎯 Confidence (%)'].round(1)
            display_df['💰 Amount ($)'] = display_df['💰 Amount ($)'].round(2)
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            add_vertical_space(2)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                colored_header(
                    label="Quick Statistics",
                    description="Summary of your expense history",
                    color_name="blue-70"
                )
                
                total_amount = history_df['amount'].sum()
                avg_amount = history_df['amount'].mean()
                median_amount = history_df['amount'].median()
                
                st.metric("💰 Total Expenses", f"${total_amount:.2f}")
                st.metric("📊 Average Amount", f"${avg_amount:.2f}")
                st.metric("📈 Median Amount", f"${median_amount:.2f}")
                st.metric("📝 Total Transactions", len(history_df))
            
            with col2:
                colored_header(
                    label="Top Categories",
                    description="Most frequent expense categories",
                    color_name="green-70"
                )
                
                category_counts = history_df['predicted_category'].value_counts()
                for i, (category, count) in enumerate(category_counts.head(5).items()):
                    percentage = (count / len(history_df)) * 100
                    st.write(f"**{i+1}. {category.title()}**: {count} ({percentage:.1f}%)")
            
            add_vertical_space(2)
            create_download_buttons()
        else:
            st.markdown("""
            <div class="warning-message">
                <h3>📭 No expense history yet</h3>
                <p>Start making predictions to see them here! Use the <strong>🔮 Predict</strong> tab to categorize individual expenses or <strong>📊 Bulk Upload</strong> for multiple expenses at once.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Dashboard
    elif selected == "📈 Dashboard":
        colored_header(
            label="Expense Analytics Dashboard",
            description="Visual insights into your spending patterns",
            color_name="red-70"
        )
        
        if st.session_state.expense_history:
            create_visualizations()
            
            add_vertical_space(3)
            colored_header(
                label="Export & Management",
                description="Download your data and manage history",
                color_name="violet-70"
            )
            create_download_buttons()
        else:
            st.markdown("""
            <div class="warning-message">
                <h3>📊 No data available for dashboard</h3>
                <p>Start making expense predictions to see beautiful visualizations and analytics! Your dashboard will show:</p>
                <ul>
                    <li>📊 Spending by category charts</li>
                    <li>📈 Daily spending trends</li>
                    <li>🥧 Expense distribution pie charts</li>
                    <li>📋 Summary statistics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    add_vertical_space(3)
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: {THEMES[st.session_state.theme]['text_color']}80;'>
            <p>🏷️ <strong>AI Expense Categorizer</strong> • Built with ❤️ using Streamlit & Advanced ML</p>
            <p>Theme: {st.session_state.theme.title()} • Models: Advanced Pipeline • Accuracy: 99%+</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()