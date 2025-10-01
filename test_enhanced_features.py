# test_enhanced_features.py - Test script for all enhanced features
import streamlit as st
import pandas as pd
import sys
import os

def test_import_functionality():
    """Test if all required imports work"""
    try:
        import joblib
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime
        import io
        from preprocess import clean_text, extract_amount_from_text
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded"""
    try:
        import joblib
        model = joblib.load("models/complete_advanced_predictor.joblib")
        print("✅ Advanced models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_bulk_csv_processing():
    """Test bulk CSV processing"""
    try:
        # Create test data
        test_data = {
            'merchant': ['McDonald\'s', 'Shell', 'Amazon'],
            'description': ['Big Mac meal', 'Gas fill up', 'Office supplies'],
            'amount': [12.99, 45.50, 89.99]
        }
        df = pd.DataFrame(test_data)
        
        # Test CSV creation
        df.to_csv('test_temp.csv', index=False)
        
        # Test CSV reading
        df_read = pd.read_csv('test_temp.csv')
        
        # Clean up
        os.remove('test_temp.csv')
        
        print("✅ CSV processing works correctly")
        return True
    except Exception as e:
        print(f"❌ CSV processing error: {e}")
        return False

def test_visualization_dependencies():
    """Test if visualization libraries work"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Test simple chart creation
        test_data = pd.DataFrame({
            'category': ['Food', 'Transport', 'Shopping'],
            'amount': [100, 50, 75]
        })
        
        fig = px.bar(test_data, x='category', y='amount')
        print("✅ Plotly visualization works correctly")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_export_functionality():
    """Test export to Excel functionality"""
    try:
        import io
        import pandas as pd
        
        # Test Excel export
        test_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Category': ['Food', 'Transport'],
            'Amount': [25.99, 15.50]
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            test_data.to_excel(writer, sheet_name='Test', index=False)
        
        print("✅ Excel export functionality works")
        return True
    except Exception as e:
        print(f"❌ Excel export error: {e}")
        return False

def main():
    print("🧪 TESTING ENHANCED APP FEATURES")
    print("=" * 50)
    
    tests = [
        ("Import Functionality", test_import_functionality),
        ("Model Loading", test_model_loading),
        ("Bulk CSV Processing", test_bulk_csv_processing),
        ("Visualization Dependencies", test_visualization_dependencies),
        ("Export Functionality", test_export_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n{'='*50}")
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced app is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()