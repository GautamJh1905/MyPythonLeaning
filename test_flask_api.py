"""
Test script for Flask Loan Prediction API
Run this after starting the Flask server
"""

import requests
import json

# API Base URL
BASE_URL = "http://127.0.0.1:5000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing Health Endpoint")
    print("="*70)

    response = requests.get(f"{BASE_URL}/health", timeout=10)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Single Prediction")
    print("="*70)

    # Test data - Good credit history
    test_data = {
        "Married": "Yes",
        "Self_Employed": "No",
        "Education": "Graduate",
        "Property_Area": "Urban",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Credit_History": 1,
        "Loan_Amount_Term": 360,
        "Dependents": "2"
    }

    print("\nInput Data:")
    print(json.dumps(test_data, indent=2))

    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'},
        timeout=10
    )

    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def test_negative_credit_history():
    """Test with negative credit history"""
    print("\n" + "="*70)
    print("Testing Negative Credit History (Should be fixed automatically)")
    print("="*70)

    # Test data with negative credit history
    test_data = {
        "Married": "No",
        "Self_Employed": "Yes",
        "Education": "Not Graduate",
        "Property_Area": "Rural",
        "ApplicantIncome": 3000,
        "CoapplicantIncome": 0,
        "LoanAmount": 100,
        "Credit_History": -5,  # NEGATIVE VALUE
        "Loan_Amount_Term": 360,
        "Dependents": "0"
    }

    print("\nInput Data (with Credit_History = -5):")
    print(json.dumps(test_data, indent=2))

    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'},
        timeout=10
    )

    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Batch Prediction")
    print("="*70)

    # Multiple applicants
    batch_data = {
        "applicants": [
            {
                "Married": "Yes",
                "Self_Employed": "No",
                "Education": "Graduate",
                "Property_Area": "Urban",
                "ApplicantIncome": 5000,
                "CoapplicantIncome": 2000,
                "LoanAmount": 150,
                "Credit_History": 1,
                "Loan_Amount_Term": 360,
                "Dependents": "2"
            },
            {
                "Married": "No",
                "Self_Employed": "Yes",
                "Education": "Not Graduate",
                "Property_Area": "Rural",
                "ApplicantIncome": 3000,
                "CoapplicantIncome": 0,
                "LoanAmount": 100,
                "Credit_History": -1,  # Negative value
                "Loan_Amount_Term": 360,
                "Dependents": "0"
            },
            {
                "Married": "Yes",
                "Self_Employed": "No",
                "Education": "Graduate",
                "Property_Area": "Semiurban",
                "ApplicantIncome": 7000,
                "CoapplicantIncome": 1500,
                "LoanAmount": 200,
                "Credit_History": 0,
                "Loan_Amount_Term": 180,
                "Dependents": "1"
            }
        ]
    }

    print(f"\nInput: {len(batch_data['applicants'])} applicants")

    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_data,
        headers={'Content-Type': 'application/json'},
        timeout=10
    )

    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def run_all_tests():
    """Run all tests"""
    try:
        print("\n" + "="*70)
        print("🧪 FLASK LOAN PREDICTION API - TEST SUITE")
        print("="*70)
        print("Make sure the Flask server is running on port 5000")
        print("="*70)

        test_health()
        test_single_prediction()
        test_negative_credit_history()
        test_batch_prediction()

        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the Flask server")
        print("Make sure to start the server first:")
        print("  python Flask_Loan_Prediction.py")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
