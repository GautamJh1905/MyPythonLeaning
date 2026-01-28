"""
Generate realistic house sales dataset for:
1. Linear Regression - Predict house price
2. Logistic Regression - Predict if house sells within 1 week
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

print("Generating house sales dataset...")

# =============================================================================
# STEP 1: Generate Base Features
# =============================================================================

# Square footage (800 - 4500 sq ft)
square_footage = np.random.randint(800, 4500, n_samples)

# Bedrooms (1-6)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[
                            0.05, 0.15, 0.35, 0.30, 0.10, 0.05])

# Bathrooms (1-4)
bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples,
                             p=[0.10, 0.15, 0.30, 0.20, 0.15, 0.07, 0.03])

# Age of house (0-50 years)
age = np.random.randint(0, 51, n_samples)

# Garage spaces (0-3)
garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.10, 0.30, 0.45, 0.15])

# Lot size (2000 - 15000 sq ft)
lot_size = np.random.randint(2000, 15000, n_samples)

# Neighborhood quality (1-10 scale)
neighborhood_rating = np.random.choice(range(1, 11), n_samples)

# House condition (1-10 scale)
condition = np.random.choice(range(1, 11), n_samples)

# Has pool (binary)
has_pool = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])

# Has renovations (binary)
renovated = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])

# Number of floors (1-3)
floors = np.random.choice([1, 2, 3], n_samples, p=[0.40, 0.50, 0.10])

# Distance to city center (km)
distance_to_center = np.random.uniform(1, 50, n_samples)

# School rating nearby (1-10)
school_rating = np.random.choice(range(1, 11), n_samples)

# =============================================================================
# STEP 2: Generate House Price (Target for Linear Regression)
# =============================================================================

# Base price calculation with realistic factors
base_price = (
    square_footage * 150 +  # $150 per sq ft
    bedrooms * 15000 +  # $15k per bedroom
    bathrooms * 12000 +  # $12k per bathroom
    garage * 20000 +  # $20k per garage space
    lot_size * 5 +  # $5 per sq ft of lot
    neighborhood_rating * 25000 +  # Neighborhood premium
    condition * 10000 +  # Condition premium
    has_pool * 35000 +  # Pool adds value
    renovated * 30000 +  # Renovation adds value
    school_rating * 8000 +  # Good schools add value
    floors * 15000 -  # Multiple floors add value
    age * 1500 -  # Age reduces value
    distance_to_center * 1000  # Distance reduces value
)

# Add some random noise (±10%)
noise = np.random.normal(0, 0.1, n_samples)
house_price = base_price * (1 + noise)

# Ensure minimum price of $50,000
house_price = np.maximum(house_price, 50000)

# Round to nearest $1000
house_price = np.round(house_price / 1000) * 1000

# =============================================================================
# STEP 3: Generate Sold Within Week (Target for Logistic Regression)
# =============================================================================

# Calculate probability of selling within a week based on factors
prob_quick_sale = (
    0.05 +  # Base probability
    # Lower price neighborhoods sell faster
    (10 - neighborhood_rating) * 0.02 +
    (10 - condition) * 0.015 +  # Better condition sells faster (inverted)
    (condition / 10) * 0.15 +  # Actually good condition helps
    (school_rating / 10) * 0.10 +  # Good schools help
    renovated * 0.10 +  # Renovations help
    has_pool * 0.08 +  # Pool helps
    (square_footage < 2000) * 0.05 +  # Smaller houses sell faster
    (bedrooms <= 3) * 0.05 +  # Moderate size sells faster
    (garage >= 2) * 0.06 +  # Good parking helps
    (house_price < 300000) * 0.15 +  # Affordable houses sell faster
    (house_price > 1000000) * -0.10 +  # Expensive houses sell slower
    (age < 10) * 0.08 +  # Newer houses sell faster
    (distance_to_center < 10) * 0.07  # Close to city sells faster
)

# Clip probabilities between 0.05 and 0.95
prob_quick_sale = np.clip(prob_quick_sale, 0.05, 0.95)

# Generate binary outcome
sold_within_week = np.random.binomial(1, prob_quick_sale, n_samples)

# =============================================================================
# STEP 4: Generate Days on Market
# =============================================================================

# If sold within week: 1-7 days, else 8-180 days
days_on_market = np.where(
    sold_within_week == 1,
    np.random.randint(1, 8, n_samples),
    np.random.randint(8, 181, n_samples)
)

# =============================================================================
# STEP 5: Generate Location Types
# =============================================================================

location_type = np.random.choice(
    ['Urban', 'Suburban', 'Rural', 'Downtown'],
    n_samples,
    p=[0.30, 0.45, 0.15, 0.10]
)

# =============================================================================
# STEP 6: Create DataFrame
# =============================================================================

df = pd.DataFrame({
    'Square_Footage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Garage_Spaces': garage,
    'Lot_Size': lot_size,
    'Floors': floors,
    'Neighborhood_Rating': neighborhood_rating,
    'Condition': condition,
    'School_Rating': school_rating,
    'Has_Pool': has_pool,
    'Renovated': renovated,
    'Location_Type': location_type,
    'Distance_To_Center_KM': np.round(distance_to_center, 2),
    'Days_On_Market': days_on_market,
    'Price': house_price.astype(int),
    'Sold_Within_Week': sold_within_week
})

# =============================================================================
# STEP 7: Add Some Missing Values (Realistic)
# =============================================================================

# Randomly set some values to NaN (about 2-3% missing data)
missing_columns = ['Lot_Size', 'Garage_Spaces',
                   'School_Rating', 'Days_On_Market']
for col in missing_columns:
    mask = np.random.random(n_samples) < 0.03
    df.loc[mask, col] = np.nan

# =============================================================================
# STEP 8: Save to CSV
# =============================================================================

df.to_csv('house_sales_data.csv', index=False)

print(f"✅ Generated {n_samples} house records")
print(f"✅ Saved to 'house_sales_data.csv'")
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n💰 Price Statistics:")
print(f"   Mean: ${df['Price'].mean():,.0f}")
print(f"   Median: ${df['Price'].median():,.0f}")
print(f"   Min: ${df['Price'].min():,.0f}")
print(f"   Max: ${df['Price'].max():,.0f}")

print(f"\n🏠 Quick Sale Statistics:")
quick_sales = df['Sold_Within_Week'].sum()
total = len(df)
print(f"   Sold within 1 week: {quick_sales} ({quick_sales/total*100:.1f}%)")
print(
    f"   Not sold within 1 week: {total-quick_sales} ({(total-quick_sales)/total*100:.1f}%)")

print(f"\n📋 Features for Linear Regression (Price Prediction):")
print(f"   - Square_Footage, Bedrooms, Bathrooms, Age, Garage_Spaces")
print(f"   - Lot_Size, Floors, Neighborhood_Rating, Condition")
print(f"   - School_Rating, Has_Pool, Renovated, Location_Type")
print(f"   - Distance_To_Center_KM")
print(f"   Target: Price")

print(f"\n📋 Features for Logistic Regression (Quick Sale Prediction):")
print(f"   - Same features as above + Price")
print(f"   Target: Sold_Within_Week (0 or 1)")

print("\n" + "="*60)
print("\n📈 Sample Data (first 5 rows):")
print(df.head())

print("\n" + "="*60)
print("\n🎯 Use Cases:")
print("   1. Linear Regression: Predict 'Price' based on house features")
print("   2. Logistic Regression: Predict 'Sold_Within_Week' (Yes/No)")
print("="*60)
