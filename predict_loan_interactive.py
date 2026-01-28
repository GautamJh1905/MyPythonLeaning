"""
Interactive Loan Prediction Script
This script takes user input and sends it to the Flask API for prediction
"""

import requests
import json


def get_user_input():
    """Get loan application details from user input"""

    print("\n" + "="*70)
    print("🏦 LOAN APPLICATION - Enter Your Details")
    print("="*70)

    print("\n📝 Please answer the following questions:")

    # Get user inputs with validation
    married = input("\n1. Are you married? (Yes/No): ").strip().capitalize()
    while married not in ["Yes", "No"]:
        print("   ⚠️  Please enter 'Yes' or 'No'")
        married = input("   Are you married? (Yes/No): ").strip().capitalize()

    self_employed = input(
        "\n2. Are you self-employed? (Yes/No): ").strip().capitalize()
    while self_employed not in ["Yes", "No"]:
        print("   ⚠️  Please enter 'Yes' or 'No'")
        self_employed = input(
            "   Are you self-employed? (Yes/No): ").strip().capitalize()

    education = input("\n3. Education level (Graduate/Not Graduate): ").strip()
    while education not in ["Graduate", "Not Graduate"]:
        print("   ⚠️  Please enter 'Graduate' or 'Not Graduate'")
        education = input(
            "   Education level (Graduate/Not Graduate): ").strip()

    property_area = input(
        "\n4. Property area (Urban/Semiurban/Rural): ").strip().capitalize()
    while property_area not in ["Urban", "Semiurban", "Rural"]:
        print("   ⚠️  Please enter 'Urban', 'Semiurban', or 'Rural'")
        property_area = input(
            "   Property area (Urban/Semiurban/Rural): ").strip().capitalize()

    while True:
        try:
            applicant_income = int(
                input("\n5. Your monthly income (e.g., 5000): "))
            if applicant_income <= 0:
                print("   ⚠️  Income must be positive")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    while True:
        try:
            coapplicant_income = int(
                input("\n6. Co-applicant monthly income (enter 0 if none): "))
            if coapplicant_income < 0:
                print("   ⚠️  Income cannot be negative")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    while True:
        try:
            loan_amount = int(
                input("\n7. Loan amount requested in thousands (e.g., 150): "))
            if loan_amount <= 0:
                print("   ⚠️  Loan amount must be positive")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    while True:
        try:
            credit_history = int(
                input("\n8. Credit history (1 for good credit, 0 for no credit history): "))
            if credit_history not in [0, 1]:
                print("   ⚠️  Please enter 0 or 1")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter 0 or 1")

    while True:
        try:
            loan_term = int(
                input("\n9. Loan term in months (e.g., 360 for 30 years): "))
            if loan_term <= 0:
                print("   ⚠️  Loan term must be positive")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    dependents = input("\n10. Number of dependents (0/1/2/3+): ").strip()
    while dependents not in ["0", "1", "2", "3+"]:
        print("   ⚠️  Please enter 0, 1, 2, or 3+")
        dependents = input("   Number of dependents (0/1/2/3+): ").strip()

    # Create applicant data dictionary
    applicant_data = {
        "Married": married,
        "Self_Employed": self_employed,
        "Education": education,
        "Property_Area": property_area,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Credit_History": credit_history,
        "Loan_Amount_Term": loan_term,
        "Dependents": dependents
    }

    return applicant_data


def send_prediction_request(applicant_data, api_url="http://127.0.0.1:5000/predict"):
    """Send prediction request to Flask API"""

    print("\n" + "="*70)
    print("📤 Sending request to Flask API...")
    print("="*70)

    try:
        response = requests.post(
            api_url,
            json=applicant_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Flask server. Make sure it's running on port 5000."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Server may be overloaded."
    except requests.exceptions.RequestException as e:
        return None, f"Error: {str(e)}"


def display_result(result):
    """Display the prediction result"""

    print("\n" + "="*70)
    print("✅ LOAN PREDICTION RESULT")
    print("="*70)

    # Main prediction
    prediction = result['prediction']
    approval_prob = result['approval_probability']
    rejection_prob = result['rejection_probability']

    if prediction == "Approved":
        print(f"\n🎉 CONGRATULATIONS! Your loan is likely to be {prediction}")
        print(f"   Approval Probability: {approval_prob:.2%}")
    else:
        print(f"\n😞 Unfortunately, your loan is likely to be {prediction}")
        print(f"   Rejection Probability: {rejection_prob:.2%}")

    # Validation report
    if result['validation_report']:
        print("\n⚠️  Validation Report:")
        for msg in result['validation_report']:
            print(f"   {msg}")

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Prediction: {prediction}")
    print(f"Approval Probability: {approval_prob:.2%}")
    print(f"Rejection Probability: {rejection_prob:.2%}")
    print("="*70)


def main():
    """Main function"""

    print("\n" + "="*70)
    print("🏦 LOAN PREDICTION SYSTEM")
    print("="*70)
    print("\nThis tool predicts your loan approval chances using Machine Learning")
    print("Make sure the Flask API server is running before proceeding.")

    while True:
        print("\n" + "="*70)

        # Get user input
        try:
            applicant_data = get_user_input()
        except KeyboardInterrupt:
            print("\n\n👋 Exiting... Goodbye!")
            break

        # Confirm data
        print("\n" + "="*70)
        print("📋 YOUR APPLICATION DETAILS")
        print("="*70)
        print(json.dumps(applicant_data, indent=2))

        confirm = input(
            "\n✓ Is this information correct? (Yes/No): ").strip().capitalize()
        if confirm != "Yes":
            print("\n🔄 Let's try again...\n")
            continue

        # Send request
        result, error = send_prediction_request(applicant_data)

        if error:
            print(f"\n❌ Error: {error}")
            print("\nTroubleshooting:")
            print("1. Make sure Flask server is running: python Flask_Loan_Prediction.py")
            print("2. Check if port 5000 is available")
            print("3. Verify the API is accessible at http://127.0.0.1:5000")
        else:
            display_result(result)

        # Ask if user wants to check another application
        print("\n" + "="*70)
        another = input(
            "\n🔁 Would you like to check another application? (Yes/No): ").strip().capitalize()
        if another != "Yes":
            print("\n👋 Thank you for using the Loan Prediction System. Goodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting... Goodbye!")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"\n❌ Unexpected error: {str(e)}")
