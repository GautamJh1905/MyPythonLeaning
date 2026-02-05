"""Simple API test to debug the preprocessing issue"""
import requests
import json

url = "http://localhost:8000/predict"

customer = {
    "CustomerID": "C100",
    "Age": 45,
    "Gender": "M",
    "City": "Mumbai",
    "AnnualIncome": 1200000.0,
    "TotalSpent": 750000.0,
    "MonthlyPurchases": 18,
    "AvgOrderValue": 10000.0,
    "AppTimeMinutes": 120.0,
    "DiscountUsage": "High",
    "PreferredShoppingTime": "Night"
}

try:
    response = requests.post(url, json=customer)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
