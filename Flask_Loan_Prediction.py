from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'loan_model.pkl'
model = None


def load_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")


# Load model on startup
load_model()


def validate_and_predict(model, entry_df, fix_invalid=True):
    """
    Validates the input data and makes predictions.

    Parameters:
    - model: Trained model pipeline
    - entry_df: DataFrame with input features
    - fix_invalid: If True, fixes invalid values; if False, raises exception

    Returns:
    - prediction, probability, validation_report
    """
    # Create a copy to avoid modifying original
    validated_df = entry_df.copy()
    validation_report = []

    # Check Credit_History values
    if 'Credit_History' in validated_df.columns:
        invalid_credit = validated_df['Credit_History'] < 0

        if invalid_credit.any():
            invalid_values = validated_df.loc[invalid_credit, 'Credit_History'].tolist(
            )

            warning_msg = f"WARNING: Found {invalid_credit.sum()} invalid Credit_History value(s): {invalid_values}"
            validation_report.append(warning_msg)

            if fix_invalid:
                # Fix: Replace negative values with 0 (no credit history)
                validated_df.loc[invalid_credit, 'Credit_History'] = 0
                fix_msg = "FIXED: Replaced negative Credit_History values with 0"
                validation_report.append(fix_msg)
            else:
                error_msg = f"ERROR: Credit_History must be 0 or 1. Found invalid values: {invalid_values}"
                validation_report.append(error_msg)
                raise ValueError(error_msg)

    # Make prediction on validated data
    prediction = model.predict(validated_df)
    prediction_proba = model.predict_proba(validated_df)

    return prediction, prediction_proba, validated_df, validation_report


@app.route("/")
def home():
    """Serve the web interface"""
    return render_template('index.html')


@app.route("/api")
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Loan Prediction API",
        "endpoints": {
            "/": "Web interface",
            "/api": "API information",
            "/predict": "POST - Make loan prediction",
            "/predict_batch": "POST - Batch predictions",
            "/health": "GET - Check API health"
        }
    })


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=['POST'])
def predict():
    """
    Predict loan approval for a single applicant

    Expected JSON format:
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
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create DataFrame from input
        input_df = pd.DataFrame([data])

        # Validate required columns
        required_cols = ['Married', 'Self_Employed', 'Education', 'Property_Area',
                         'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                         'Credit_History', 'Loan_Amount_Term', 'Dependents']

        missing_cols = [
            col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {missing_cols}"}), 400

        # Make prediction with validation
        prediction, proba, _, report = validate_and_predict(
            model, input_df, fix_invalid=True
        )

        # Prepare response
        result = {
            "prediction": "Approved" if prediction[0] == 1 else "Rejected",
            "prediction_value": int(prediction[0]),
            "approval_probability": float(proba[0][1]),
            "rejection_probability": float(proba[0][0]),
            "input_data": data,
            "validation_report": report
        }

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (AttributeError, KeyError, TypeError) as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict_batch", methods=['POST'])
def predict_batch():
    """
    Predict loan approval for multiple applicants

    Expected JSON format:
    {
        "applicants": [
            {
                "Married": "Yes",
                "Self_Employed": "No",
                ...
            },
            {
                "Married": "No",
                "Self_Employed": "Yes",
                ...
            }
        ]
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()

        if not data or 'applicants' not in data:
            return jsonify({"error": "No applicants data provided"}), 400

        applicants = data['applicants']

        if not applicants:
            return jsonify({"error": "Empty applicants list"}), 400

        # Create DataFrame from input
        input_df = pd.DataFrame(applicants)

        # Make predictions with validation
        predictions, proba, _, report = validate_and_predict(
            model, input_df, fix_invalid=True
        )

        # Prepare response
        results = []
        for i in range(len(predictions)):
            results.append({
                "applicant_index": i,
                "prediction": "Approved" if predictions[i] == 1 else "Rejected",
                "prediction_value": int(predictions[i]),
                "approval_probability": float(proba[i][1]),
                "rejection_probability": float(proba[i][0])
            })

        response = {
            "total_applicants": len(predictions),
            "approved_count": int(sum(predictions)),
            "rejected_count": int(len(predictions) - sum(predictions)),
            "validation_report": report,
            "results": results
        }

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (AttributeError, KeyError, TypeError) as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting Loan Prediction API - Browser Interface")
    print("="*70)
    print(f"Model Status: {'Loaded' if model else 'Not Loaded'}")
    print("\nAvailable Endpoints:")
    print("  - GET  /           : Web Interface (HTML Form)")
    print("  - GET  /api        : API information")
    print("  - GET  /health     : Health check")
    print("  - POST /predict    : Single prediction")
    print("  - POST /predict_batch : Batch predictions")
    print("="*70)
    print("🌐 Open in browser: http://127.0.0.1:5000")
    print("="*70)
    app.run(debug=True, port=5000, use_reloader=False)
