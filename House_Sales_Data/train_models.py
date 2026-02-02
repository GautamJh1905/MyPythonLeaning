"""
Train Linear and Logistic Regression models for house sales predictions
- Linear Regression: Predict house price
- Logistic Regression: Predict if sold within a week
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading house sales data...")
df = pd.read_csv('house_sales_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Check for missing values
print("Missing values:")
print(df.isnull().sum())
print()

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical features
print("Encoding categorical features...")
le = LabelEncoder()
if 'Location_Type' in df.columns:
    df['Location_Type_Encoded'] = le.fit_transform(df['Location_Type'])
    # Save the label encoder for later use
    with open('location_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

# Define features (exclude target variables and original categorical column)
feature_columns = [col for col in df.columns if col not in [
    'Price', 'Sold_Within_Week', 'Location_Type']]
X = df[feature_columns]

print(f"Features used: {feature_columns}\n")

# ========================================
# PART 1: LINEAR REGRESSION FOR PRICE
# ========================================
print("="*60)
print("TRAINING LINEAR REGRESSION MODEL FOR PRICE PREDICTION")
print("="*60)

y_price = df['Price']

# Split the data
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# Scale features for better performance
scaler_price = StandardScaler()
X_train_price_scaled = scaler_price.fit_transform(X_train_price)
X_test_price_scaled = scaler_price.transform(X_test_price)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_price_scaled, y_train_price)

# Cross-validation
cv_scores_price = cross_val_score(lr_model, X_train_price_scaled, y_train_price, cv=5,
                                  scoring='r2')
print(f"\nCross-Validation R² Scores: {cv_scores_price}")
print(
    f"Mean CV R² Score: {cv_scores_price.mean():.4f} (+/- {cv_scores_price.std():.4f})")

# Evaluate on test set
y_pred_price = lr_model.predict(X_test_price_scaled)
rmse_price = np.sqrt(mean_squared_error(y_test_price, y_pred_price))
r2_price = r2_score(y_test_price, y_pred_price)

print(f"\nTest Set Performance:")
print(f"R² Score: {r2_price:.4f}")
print(f"RMSE: ${rmse_price:,.2f}")

# Save the price prediction model and scaler
with open('price_model.pkl', 'wb') as f:
    pickle.dump({
        'model': lr_model,
        'scaler': scaler_price,
        'features': feature_columns
    }, f)
print("\n✓ Price model saved as 'price_model.pkl'")

# ========================================
# PART 2: LOGISTIC REGRESSION FOR SOLD_WITHIN_WEEK
# ========================================
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION MODEL FOR SOLD_WITHIN_WEEK")
print("="*60)

y_sold = df['Sold_Within_Week']

# Split the data
X_train_sold, X_test_sold, y_train_sold, y_test_sold = train_test_split(
    X, y_sold, test_size=0.2, random_state=42, stratify=y_sold
)

# Scale features
scaler_sold = StandardScaler()
X_train_sold_scaled = scaler_sold.fit_transform(X_train_sold)
X_test_sold_scaled = scaler_sold.transform(X_test_sold)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_sold_scaled, y_train_sold)

# Cross-validation
cv_scores_sold = cross_val_score(log_model, X_train_sold_scaled, y_train_sold, cv=5,
                                 scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores: {cv_scores_sold}")
print(
    f"Mean CV Accuracy: {cv_scores_sold.mean():.4f} (+/- {cv_scores_sold.std():.4f})")

# Evaluate on test set
y_pred_sold = log_model.predict(X_test_sold_scaled)
accuracy_sold = accuracy_score(y_test_sold, y_pred_sold)

print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy_sold:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_sold, y_pred_sold,
      target_names=['Not Sold', 'Sold']))

# Crosstab (confusion matrix)
print("\nCrosstab (Confusion Matrix):")
crosstab = pd.crosstab(y_test_sold, y_pred_sold,
                       rownames=['Actual'], colnames=['Predicted'],
                       margins=True)
print(crosstab)

# Save the sold prediction model and scaler
with open('sold_model.pkl', 'wb') as f:
    pickle.dump({
        'model': log_model,
        'scaler': scaler_sold,
        'features': feature_columns
    }, f)
print("\n✓ Sold model saved as 'sold_model.pkl'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print("Created files:")
print("  - price_model.pkl (Linear Regression for price prediction)")
print("  - sold_model.pkl (Logistic Regression for sold prediction)")
print("  - location_encoder.pkl (Label encoder for Location_Type)")
