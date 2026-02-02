# Data manipulation and analysis
import numpy as np
import pandas as pd

# Machine Learning models
from sklearn.linear_model import LinearRegression, LogisticRegression

# Data preprocessing and splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Metrics for evaluation
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,

    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Load the house sales data
df = pd.read_csv('house_sales_data.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
