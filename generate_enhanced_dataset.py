import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define comprehensive categories and their typical merchants
categories_data = {
    'food': {
        'merchants': {
            'McDonald\'s': ['Big Mac meal', 'quarter pounder combo', 'chicken nuggets', 'breakfast McMuffin', 'McFlurry dessert'],
            'Starbucks': ['grande latte', 'cappuccino and pastry', 'iced coffee', 'frappuccino', 'breakfast sandwich'],
            'KFC': ['family bucket meal', 'chicken tenders', 'zinger burger combo', 'popcorn chicken', 'colonel\'s meal'],
            'Pizza Hut': ['large pepperoni pizza', 'medium cheese pizza', 'personal pan pizza', 'breadsticks and wings', 'pizza delivery'],
            'Domino\'s': ['two medium pizzas', 'large supreme pizza', 'chicken wings', 'pizza pickup order', 'garlic bread'],
            'Subway': ['foot-long sandwich', '6-inch turkey sub', 'Italian BMT', 'veggie delite', 'cookies and drink'],
            'Chipotle': ['burrito bowl', 'chicken burrito', 'carnitas tacos', 'guacamole and chips', 'quesadilla meal'],
            'Whole Foods': ['organic groceries', 'fresh produce', 'weekly shopping', 'healthy snacks', 'prepared meals'],
            'Trader Joe\'s': ['frozen meals', 'wine and cheese', 'organic vegetables', 'specialty items', 'grocery run']
        },
        'amount_ranges': {
            'McDonald\'s': (6, 25), 'Starbucks': (4, 20), 'KFC': (8, 40), 'Pizza Hut': (12, 60),
            'Domino\'s': (15, 50), 'Subway': (6, 25), 'Chipotle': (8, 30), 'Whole Foods': (20, 200),
            'Trader Joe\'s': (15, 150)
        }
    },
    'transport': {
        'merchants': {
            'Uber': ['ride to airport', 'trip to downtown', 'ride home', 'grocery store trip', 'late night ride'],
            'Lyft': ['work commute', 'ride to meeting', 'airport pickup', 'weekend trip', 'shared ride'],
            'Shell': ['gas tank fill-up', 'fuel for road trip', 'premium gasoline', 'car wash service', 'convenience items'],
            'BP': ['regular gas fill', 'diesel fuel', 'snacks and gas', 'fuel stop', 'highway gas station'],
            'Chevron': ['gasoline purchase', 'full tank gas', 'fuel and coffee', 'car maintenance', 'oil change service'],
            'Metro Transit': ['monthly bus pass', 'daily transit fare', 'subway ticket', 'public transport', 'weekly pass'],
            'Parking Authority': ['downtown parking', 'airport parking', 'meter parking', 'monthly parking pass', 'event parking']
        },
        'amount_ranges': {
            'Uber': (8, 80), 'Lyft': (7, 75), 'Shell': (25, 120), 'BP': (20, 100),
            'Chevron': (30, 130), 'Metro Transit': (2, 150), 'Parking Authority': (5, 200)
        }
    },
    'shopping': {
        'merchants': {
            'Amazon': ['laptop accessories', 'books and supplies', 'household essentials', 'electronics gadget', 'clothing order'],
            'Walmart': ['weekly groceries', 'household supplies', 'pharmacy items', 'clothing basics', 'garden supplies'],
            'Target': ['home decor items', 'personal care', 'kitchen supplies', 'seasonal items', 'gift cards'],
            'Best Buy': ['phone charger', 'computer mouse', 'headphones', 'video game', 'tech accessories'],
            'Costco': ['bulk groceries', 'household bulk items', 'warehouse shopping', 'family supplies', 'bulk snacks'],
            'Macy\'s': ['dress shirt', 'formal wear', 'jewelry gift', 'cosmetics', 'designer clothes'],
            'H&M': ['casual clothing', 'spring fashion', 'jeans and tops', 'accessories', 'seasonal wear'],
            'Home Depot': ['hardware supplies', 'garden tools', 'paint and brushes', 'home repair items', 'lighting fixtures']
        },
        'amount_ranges': {
            'Amazon': (15, 300), 'Walmart': (25, 200), 'Target': (20, 150), 'Best Buy': (30, 800),
            'Costco': (50, 400), 'Macy\'s': (40, 250), 'H&M': (25, 120), 'Home Depot': (20, 350)
        }
    },
    'entertainment': {
        'merchants': {
            'Netflix': ['monthly streaming', 'premium subscription', 'family plan', 'video streaming service', 'monthly renewal'],
            'Spotify': ['music streaming', 'premium subscription', 'monthly music plan', 'ad-free music', 'family music plan'],
            'Disney+': ['streaming subscription', 'monthly Disney plan', 'family entertainment', 'streaming service', 'annual subscription'],
            'AMC Theaters': ['movie tickets', 'evening show tickets', 'IMAX movie', 'popcorn and movie', 'weekend movie'],
            'Steam': ['video game purchase', 'game download', 'gaming software', 'indie game', 'AAA game title'],
            'PlayStation Store': ['game download', 'DLC purchase', 'PS Plus subscription', 'digital game', 'gaming content'],
            'Concert Venue': ['concert tickets', 'live music event', 'festival pass', 'band performance', 'music concert']
        },
        'amount_ranges': {
            'Netflix': (10, 20), 'Spotify': (5, 15), 'Disney+': (8, 15), 'AMC Theaters': (12, 50),
            'Steam': (5, 80), 'PlayStation Store': (10, 70), 'Concert Venue': (30, 200)
        }
    },
    'technology': {
        'merchants': {
            'Apple Store': ['iPhone purchase', 'MacBook laptop', 'iPad tablet', 'AirPods headphones', 'charging accessories'],
            'Microsoft Store': ['Surface laptop', 'Office 365 subscription', 'Xbox controller', 'Windows software', 'Microsoft hardware'],
            'Google Play': ['app purchase', 'mobile game', 'premium app', 'in-app purchase', 'digital content'],
            'Samsung': ['Galaxy smartphone', 'tablet device', 'smartwatch', 'phone accessories', 'tech gadget'],
            'Best Buy': ['laptop computer', 'wireless headphones', 'phone charger', 'computer mouse', 'tech accessories'],
            'Amazon Tech': ['electronics order', 'computer parts', 'smart home device', 'tech gadgets', 'cables and adapters']
        },
        'amount_ranges': {
            'Apple Store': (50, 1500), 'Microsoft Store': (30, 1200), 'Google Play': (1, 50),
            'Samsung': (100, 1000), 'Best Buy': (20, 800), 'Amazon Tech': (15, 500)
        }
    },
    'utilities': {
        'merchants': {
            'ComEd': ['monthly electric bill', 'electricity usage', 'power bill payment', 'energy charges', 'electric service'],
            'Peoples Gas': ['natural gas bill', 'heating bill', 'gas service charge', 'monthly gas usage', 'winter heating'],
            'Water Department': ['water and sewer bill', 'monthly water service', 'utility water bill', 'water usage charge', 'municipal water'],
            'Xfinity': ['internet service', 'cable and internet', 'monthly internet bill', 'broadband service', 'wifi service'],
            'Verizon': ['cell phone bill', 'wireless service', 'monthly phone plan', 'mobile service', 'phone and data'],
            'AT&T': ['internet and phone', 'fiber internet', 'wireless bill', 'monthly service', 'telecom services'],
            'Waste Management': ['garbage collection', 'waste pickup service', 'monthly trash bill', 'recycling service', 'waste disposal']
        },
        'amount_ranges': {
            'ComEd': (60, 250), 'Peoples Gas': (40, 200), 'Water Department': (30, 150),
            'Xfinity': (70, 180), 'Verizon': (50, 200), 'AT&T': (60, 220), 'Waste Management': (25, 80)
        }
    },
    'healthcare': {
        'merchants': {
            'CVS Pharmacy': ['prescription refill', 'flu medication', 'vitamins and supplements', 'over-the-counter medicine', 'pharmacy pickup'],
            'Walgreens': ['prescription drugs', 'allergy medicine', 'pain relievers', 'medical supplies', 'health products'],
            'Family Clinic': ['doctor visit', 'annual checkup', 'medical consultation', 'routine examination', 'health screening'],
            'Dental Office': ['teeth cleaning', 'dental checkup', 'cavity filling', 'dental examination', 'oral care'],
            'Eye Care Center': ['eye examination', 'vision test', 'contact lenses', 'eyeglasses', 'optical services'],
            'Physical Therapy': ['therapy session', 'rehabilitation', 'physical treatment', 'injury recovery', 'therapeutic care']
        },
        'amount_ranges': {
            'CVS Pharmacy': (10, 150), 'Walgreens': (8, 120), 'Family Clinic': (100, 400),
            'Dental Office': (80, 500), 'Eye Care Center': (50, 300), 'Physical Therapy': (75, 200)
        }
    },
    'rent': {
        'merchants': {
            'Property Management': ['monthly apartment rent', 'rental payment', 'housing payment', 'lease payment', 'apartment fee'],
            'Landlord Services': ['monthly rent', 'property rent', 'residential rent', 'housing cost', 'rental fee'],
            'Apartment Complex': ['unit rental payment', 'monthly housing', 'apartment rent', 'complex fees', 'residential payment'],
            'Real Estate Rental': ['property rental', 'housing rental', 'monthly lease', 'rental property', 'real estate rent'],
            'Housing Authority': ['subsidized rent', 'public housing', 'assisted housing', 'housing payment', 'rental assistance']
        },
        'amount_ranges': {
            'Property Management': (800, 2500), 'Landlord Services': (700, 2200), 'Apartment Complex': (900, 2800),
            'Real Estate Rental': (1000, 3200), 'Housing Authority': (400, 1200)
        }
    },
    'travel': {
        'merchants': {
            'Delta Airlines': ['flight to New York', 'domestic flight', 'business trip flight', 'vacation flight', 'roundtrip ticket'],
            'Southwest Airlines': ['flight booking', 'travel to Chicago', 'weekend trip flight', 'family vacation flight', 'business travel'],
            'Marriott Hotel': ['hotel accommodation', '3 nights stay', 'business trip hotel', 'weekend getaway', 'vacation lodging'],
            'Hilton Hotels': ['hotel reservation', 'conference hotel', 'business stay', 'family vacation hotel', 'travel accommodation'],
            'Airbnb': ['vacation rental', 'weekend rental', 'travel accommodation', 'short-term rental', 'holiday rental'],
            'Enterprise Rental': ['car rental', 'vacation car rental', 'business trip rental', 'weekend car rental', 'travel vehicle'],
            'Expedia': ['travel booking', 'vacation package', 'flight and hotel', 'travel reservation', 'trip booking']
        },
        'amount_ranges': {
            'Delta Airlines': (150, 800), 'Southwest Airlines': (100, 600), 'Marriott Hotel': (120, 400),
            'Hilton Hotels': (100, 350), 'Airbnb': (80, 300), 'Enterprise Rental': (50, 200), 'Expedia': (200, 1200)
        }
    },
    'education': {
        'merchants': {
            'University': ['semester tuition', 'course registration', 'student fees', 'education payment', 'academic fees'],
            'Coursera': ['online course', 'certification program', 'specialization course', 'skill development', 'professional course'],
            'Udemy': ['programming course', 'business course', 'skill training', 'online learning', 'professional development'],
            'Amazon Books': ['textbooks', 'study materials', 'educational books', 'reference books', 'academic resources'],
            'Campus Bookstore': ['textbook purchase', 'school supplies', 'study materials', 'academic books', 'course materials'],
            'Khan Academy': ['educational subscription', 'learning platform', 'skill building', 'academic support', 'study resources']
        },
        'amount_ranges': {
            'University': (500, 8000), 'Coursera': (30, 200), 'Udemy': (10, 150),
            'Amazon Books': (25, 400), 'Campus Bookstore': (50, 600), 'Khan Academy': (0, 50)
        }
    },
    'insurance': {
        'merchants': {
            'Blue Cross Blue Shield': ['health insurance premium', 'medical coverage', 'monthly health plan', 'healthcare premium', 'insurance payment'],
            'State Farm': ['auto insurance', 'car insurance premium', 'vehicle coverage', 'monthly car insurance', 'auto coverage'],
            'Allstate': ['home insurance', 'property insurance', 'homeowners coverage', 'house insurance', 'property premium'],
            'Progressive': ['auto insurance premium', 'car coverage', 'vehicle insurance', 'driving insurance', 'monthly auto premium'],
            'Liberty Mutual': ['life insurance premium', 'term life insurance', 'insurance coverage', 'life coverage', 'protection plan'],
            'MetLife': ['dental insurance', 'vision insurance', 'supplemental coverage', 'additional insurance', 'healthcare supplement']
        },
        'amount_ranges': {
            'Blue Cross Blue Shield': (200, 600), 'State Farm': (80, 300), 'Allstate': (100, 400),
            'Progressive': (75, 250), 'Liberty Mutual': (50, 200), 'MetLife': (30, 150)
        }
    },
    'fitness': {
        'merchants': {
            'Planet Fitness': ['monthly gym membership', 'fitness membership', 'gym access', 'workout membership', 'health club'],
            'LA Fitness': ['gym membership fee', 'fitness center access', 'monthly gym dues', 'workout facility', 'health club membership'],
            'Yoga Studio': ['yoga classes', 'meditation class', 'wellness session', 'yoga membership', 'mindfulness class'],
            'Personal Trainer': ['training session', 'fitness coaching', 'workout training', 'personal fitness', 'exercise coaching'],
            'Nike Store': ['workout clothes', 'athletic shoes', 'fitness gear', 'sports apparel', 'exercise equipment'],
            'Sports Authority': ['fitness equipment', 'exercise gear', 'workout accessories', 'sports equipment', 'athletic supplies']
        },
        'amount_ranges': {
            'Planet Fitness': (10, 50), 'LA Fitness': (25, 80), 'Yoga Studio': (15, 40),
            'Personal Trainer': (50, 150), 'Nike Store': (30, 200), 'Sports Authority': (20, 300)
        }
    }
}

def generate_enhanced_dataset(num_records=10000):
    """Generate a comprehensive expense dataset with realistic merchant-description-amount combinations"""
    
    records = []
    
    # Calculate number of records per category (with some variation)
    categories = list(categories_data.keys())
    base_per_category = num_records // len(categories)
    
    for i, category in enumerate(categories):
        # Add some variation in category distribution
        if category in ['food', 'transport', 'shopping']:  # More common categories
            category_count = int(base_per_category * 1.3)
        elif category in ['rent', 'utilities']:  # Regular monthly expenses
            category_count = int(base_per_category * 1.1)
        else:
            category_count = base_per_category
        
        category_info = categories_data[category]
        
        for j in range(category_count):
            # Generate expense record
            expense_id = f"EXP{len(records) + 1:05d}"
            
            # For categories with realistic merchant-description mapping
            if isinstance(category_info.get('merchants'), dict):
                # Select random merchant and corresponding realistic description
                merchant = random.choice(list(category_info['merchants'].keys()))
                description = random.choice(category_info['merchants'][merchant])
                
                # Use merchant-specific amount range if available
                if 'amount_ranges' in category_info and merchant in category_info['amount_ranges']:
                    min_amt, max_amt = category_info['amount_ranges'][merchant]
                else:
                    min_amt, max_amt = (10, 200)  # fallback range
                    
                amount = round(np.random.uniform(min_amt, max_amt), 2)
                
            else:
                # Fallback for older format
                merchant = random.choice(category_info['merchants'] if isinstance(category_info['merchants'], list) else list(category_info['merchants'].keys()))
                description = random.choice(category_info['descriptions'])
                min_amt, max_amt = category_info['amount_range']
                amount = round(np.random.uniform(min_amt, max_amt), 2)
            
            # Special handling for certain categories
            if category == 'rent':
                # Rent is usually consistent monthly with realistic ranges
                base_rent = random.choice([800, 950, 1100, 1250, 1400, 1600, 1800, 2000, 2200])
                amount = round(base_rent + np.random.normal(0, 100), 2)
                amount = max(600, min(3500, amount))
            elif category == 'utilities':
                # Utilities have seasonal variation but reasonable ranges
                if 'electric' in description.lower():
                    amount = round(np.random.gamma(3, 35), 2)  # Typically 60-180
                elif 'gas' in description.lower():
                    amount = round(np.random.gamma(2, 40), 2)  # Typically 40-160
                elif 'water' in description.lower():
                    amount = round(np.random.gamma(2, 25), 2)  # Typically 30-100
                amount = max(20, min(350, amount))
            
            records.append({
                'expense_id': expense_id,
                'amount': amount,
                'merchant': merchant,
                'description': description,
                'category': category
            })
    
    # Shuffle the records to mix categories
    random.shuffle(records)
    
    # Trim to exact number requested
    records = records[:num_records]
    
    # Update expense IDs to be sequential
    for i, record in enumerate(records, 1):
        record['expense_id'] = f"EXP{i:05d}"
    
    return pd.DataFrame(records)

# Generate the enhanced dataset
print("Generating enhanced expense dataset...")
enhanced_df = generate_enhanced_dataset(10000)

print(f"Generated dataset shape: {enhanced_df.shape}")
print(f"\nCategory distribution:")
print(enhanced_df['category'].value_counts().sort_index())

print(f"\nAmount statistics by category:")
print(enhanced_df.groupby('category')['amount'].agg(['count', 'mean', 'std', 'min', 'max']).round(2))

# Save the enhanced dataset
enhanced_df.to_csv('enhanced_expense_dataset.csv', index=False)
print(f"\nEnhanced dataset saved as 'enhanced_expense_dataset.csv'")

# Display sample records
print(f"\nSample records:")
print(enhanced_df.head(10))