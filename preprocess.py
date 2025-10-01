# preprocess.py
import re
import pandas as pd

def clean_text(s: str) -> str:
    """Lowercase, remove non-alphanum except spaces, collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_amount_from_text(s: str) -> float:
    """Try to find first number in text and return as float. Return 0 if none."""
    m = re.search(r'([0-9]+(?:[,\.\s][0-9]+)*)', str(s))
    if m:
        num = m.group(1).replace(',', '').replace(' ', '')
        try:
            return float(num)
        except:
            return 0.0
    return 0.0

def prepare_dataframe(df: pd.DataFrame,
                      merchant_col: str = 'merchant',
                      desc_col: str = 'description',
                      amount_col: str = 'amount') -> pd.DataFrame:
    """
    Create a cleaned 'text' column (merchant + description),
    convert amount to numeric, and return cleaned df.
    Enhanced version with better text processing for more categories.
    """
    df = df.copy()
    # fill NaNs
    df[merchant_col] = df.get(merchant_col, '').fillna('').astype(str)
    df[desc_col] = df.get(desc_col, '').fillna('').astype(str)
    
    # Enhanced text combination with better formatting
    df['text'] = (df[merchant_col] + ' ' + df[desc_col]).str.strip()
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty text entries
    df = df[df['text'].str.len() > 0]
    
    # Enhanced amount processing
    if amount_col in df.columns:
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0)
        # Remove unrealistic amounts (negative or extremely high)
        df = df[(df[amount_col] >= 0) & (df[amount_col] <= 10000)]
    else:
        df[amount_col] = 0.0
    
    return df

def get_category_keywords():
    """
    Return a dictionary of category keywords for enhanced classification.
    Useful for feature engineering or rule-based preprocessing.
    """
    return {
        'food': ['restaurant', 'cafe', 'meal', 'breakfast', 'lunch', 'dinner', 'coffee', 
                'pizza', 'burger', 'food', 'groceries', 'supermarket'],
        'transport': ['uber', 'lyft', 'taxi', 'gas', 'fuel', 'parking', 'metro', 'bus', 
                     'car', 'vehicle', 'transportation'],
        'shopping': ['store', 'mall', 'retail', 'clothes', 'clothing', 'electronics', 
                    'purchase', 'buy', 'shopping'],
        'entertainment': ['movie', 'cinema', 'netflix', 'spotify', 'game', 'concert', 
                         'theater', 'entertainment', 'streaming'],
        'technology': ['apple', 'microsoft', 'software', 'app', 'tech', 'computer', 
                      'phone', 'laptop', 'gadget'],
        'utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'utility', 'bill'],
        'healthcare': ['hospital', 'doctor', 'medical', 'pharmacy', 'health', 'clinic', 
                      'dental', 'medicine'],
        'rent': ['rent', 'apartment', 'housing', 'lease', 'property', 'landlord'],
        'travel': ['hotel', 'flight', 'airline', 'vacation', 'travel', 'booking', 'trip'],
        'education': ['school', 'university', 'course', 'education', 'tuition', 'learning'],
        'insurance': ['insurance', 'premium', 'coverage', 'policy'],
        'fitness': ['gym', 'fitness', 'workout', 'sports', 'exercise', 'health club']
    }

def extract_category_features(text: str) -> dict:
    """
    Extract category-specific features from text.
    Returns a dictionary with feature counts for each category.
    """
    text_lower = text.lower()
    category_keywords = get_category_keywords()
    
    features = {}
    for category, keywords in category_keywords.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        features[f'{category}_keywords'] = count
    
    return features
