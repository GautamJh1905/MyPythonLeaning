import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers to generate
n_customers = 5000

# Name lists
first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer',
               'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Susan',
               'Richard', 'Jessica', 'Joseph', 'Sarah', 'Thomas', 'Karen',
               'Charles', 'Nancy', 'Christopher', 'Lisa', 'Daniel', 'Margaret',
               'Matthew', 'Betty', 'Anthony', 'Dorothy', 'Mark', 'Sandra',
               'Donald', 'Ashley', 'Steven', 'Kimberly', 'Paul', 'Emily']

last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
              'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
              'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore',
              'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White',
              'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson',
              'Walker', 'Young', 'Allen', 'King', 'Wright', 'Scott']

# Generate random data
data = {
    'customer_id': range(1, n_customers + 1),
    'first_name': np.random.choice(first_names, n_customers),
    'last_name': np.random.choice(last_names, n_customers),
    'age': np.random.randint(18, 70, n_customers),
    'annual_income': np.random.uniform(20000, 150000, n_customers),
    'spending_score': np.random.randint(1, 100, n_customers),
}

# Create DataFrame
df = pd.DataFrame(data)

# Round annual income to 2 decimal places
df['annual_income'] = df['annual_income'].round(2)

# Save to CSV
output_file = 'customer_data_5000.csv'
df.to_csv(output_file, index=False)

print(f"Dataset generated successfully!")
print(f"File: {output_file}")
print(f"Total records: {len(df)}")
print("\nFirst few rows:")
print(df.head(10))
print("\nDataset statistics:")
print(df.describe())
