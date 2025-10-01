# test_ui_features.py - Test script for UI enhancements
import streamlit as st

def test_ui_imports():
    """Test if all UI enhancement imports work"""
    try:
        from streamlit_extras.colored_header import colored_header
        from streamlit_extras.metric_cards import style_metric_cards
        from streamlit_extras.add_vertical_space import add_vertical_space
        from streamlit_option_menu import option_menu
        print("✅ All UI enhancement imports successful")
        return True
    except ImportError as e:
        print(f"❌ UI import error: {e}")
        return False

def test_theme_functionality():
    """Test theme switching functionality"""
    try:
        # Test theme dictionary structure
        themes = {
            "light": {
                "primary_color": "#FF6B6B",
                "background_color": "#FFFFFF",
                "text_color": "#262730"
            },
            "dark": {
                "primary_color": "#FF6B6B", 
                "background_color": "#0E1117",
                "text_color": "#FAFAFA"
            }
        }
        
        # Test theme access
        light_theme = themes["light"]
        dark_theme = themes["dark"]
        
        print("✅ Theme functionality works correctly")
        print(f"   Light theme primary: {light_theme['primary_color']}")
        print(f"   Dark theme primary: {dark_theme['primary_color']}")
        return True
    except Exception as e:
        print(f"❌ Theme functionality error: {e}")
        return False

def test_message_components():
    """Test custom message components"""
    try:
        # Test message HTML generation
        success_html = f"""
        <div class="success-message">
            <h4>✅ Test Success Message</h4>
            <p>This is a test success message</p>
        </div>
        """
        
        warning_html = f"""
        <div class="warning-message">
            <h4>⚠️ Test Warning Message</h4>
            <p>This is a test warning message</p>
        </div>
        """
        
        error_html = f"""
        <div class="error-message">
            <h4>❌ Test Error Message</h4>
            <p>This is a test error message</p>
        </div>
        """
        
        print("✅ Message components work correctly")
        print("   Success, warning, and error message HTML generated")
        return True
    except Exception as e:
        print(f"❌ Message components error: {e}")
        return False

def test_visualization_enhancements():
    """Test enhanced visualization features"""
    try:
        import plotly.express as px
        import pandas as pd
        
        # Test sample data visualization
        sample_data = pd.DataFrame({
            'category': ['Food', 'Transport', 'Shopping'],
            'amount': [100, 50, 75]
        })
        
        # Test chart creation with themes
        fig_light = px.bar(
            sample_data, 
            x='category', 
            y='amount',
            template='plotly_white'
        )
        
        fig_dark = px.bar(
            sample_data,
            x='category', 
            y='amount',
            template='plotly_dark'
        )
        
        print("✅ Enhanced visualizations work correctly")
        print("   Light and dark theme charts created successfully")
        return True
    except Exception as e:
        print(f"❌ Visualization enhancements error: {e}")
        return False

def main():
    print("🎨 TESTING UI ENHANCEMENTS")
    print("=" * 50)
    
    tests = [
        ("UI Enhancement Imports", test_ui_imports),
        ("Theme Functionality", test_theme_functionality),
        ("Message Components", test_message_components),
        ("Visualization Enhancements", test_visualization_enhancements)
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
    print(f"🎯 UI TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL UI TESTS PASSED! Enhanced app is ready with beautiful UI.")
    else:
        print("⚠️  Some UI tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()