"""
Test script for Customer Segmentation API
Tests all endpoints with sample data
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_root():
    """Test root endpoint"""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check Endpoint")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_segments():
    """Test segments information endpoint"""
    print_section("Testing Segments Endpoint")
    response = requests.get(f"{BASE_URL}/segments")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total Clusters: {data['total_clusters']}")
    for cluster_id, info in data['segments'].items():
        print(f"\nCluster {cluster_id}: {info['name']}")
        print(f"  Description: {info['description']}")
    return response.status_code == 200


def test_prediction(customer_data, test_name):
    """Test prediction endpoint with sample data"""
    print_section(f"Testing Prediction: {test_name}")
    print(f"Input Data:")
    print(json.dumps(customer_data, indent=2))

    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction Successful!")
        print(f"  Cluster: {result['cluster']}")
        print(f"  Segment: {result['segment_name']}")
        print(f"  Description: {result['segment_description']}")
        print(f"  Confidence: {result['confidence_score']:.2%}")
        print(f"\n  Recommended Offers:")
        for offer in result['recommended_offers']:
            print(f"    • {offer}")
        return True
    else:
        print(f"\n❌ Prediction Failed!")
        print(f"  Error: {response.json()}")
        return False


def main():
    """Run all tests"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "CUSTOMER SEGMENTATION API TESTS" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    results = []

    # Test 1: Root endpoint
    results.append(("Root Endpoint", test_root()))

    # Test 2: Health check
    results.append(("Health Check", test_health()))

    # Test 3: Segments info
    results.append(("Segments Info", test_segments()))

    # Test 4: Sample customer predictions

    # Sample 1: High-value customer (likely Cluster 1)
    high_value_customer = {
        "CustomerID": "C001",
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
    results.append(("High-Value Customer",
                   test_prediction(high_value_customer, "High-Value Customer")))

    # Sample 2: Value-seeking customer (likely Cluster 2)
    value_customer = {
        "CustomerID": "C002",
        "Age": 35,
        "Gender": "F",
        "City": "Delhi",
        "AnnualIncome": 600000.0,
        "TotalSpent": 250000.0,
        "MonthlyPurchases": 9,
        "AvgOrderValue": 3500.0,
        "AppTimeMinutes": 70.0,
        "DiscountUsage": "Medium",
        "PreferredShoppingTime": "Day"
    }
    results.append(("Value-Seeking Customer",
                   test_prediction(value_customer, "Value-Seeking Customer")))

    # Sample 3: Price-sensitive customer (likely Cluster 0)
    price_sensitive_customer = {
        "CustomerID": "C003",
        "Age": 30,
        "Gender": "F",
        "City": "Bangalore",
        "AnnualIncome": 250000.0,
        "TotalSpent": 50000.0,
        "MonthlyPurchases": 2,
        "AvgOrderValue": 1500.0,
        "AppTimeMinutes": 20.0,
        "DiscountUsage": "Low",
        "PreferredShoppingTime": "Day"
    }
    results.append(("Price-Sensitive Customer",
                   test_prediction(price_sensitive_customer, "Price-Sensitive Customer")))

    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("   Make sure the API is running with: python app.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
