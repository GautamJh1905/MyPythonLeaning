# =============================================================================
# FRONTEND LAYER - Streamlit UI
# =============================================================================
"""
This is the frontend layer that provides the user interface.
It calls the backend API for all operations (predictions, database, etc.)
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

# =============================================================================
# STEP 1: Configuration
# =============================================================================

# Backend API URL
API_URL = "http://localhost:8000"

# Configure Streamlit page
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STEP 2: Custom CSS Styling
# =============================================================================

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# STEP 3: API Helper Functions
# =============================================================================


def check_api_connection():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information from backend"""
    try:
        response = requests.get(f"{API_URL}/model-info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def make_prediction(application_data: Dict):
    """Call backend API to make prediction"""
    try:
        response = requests.post(f"{API_URL}/predict", json=application_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def get_statistics():
    """Get prediction statistics from backend"""
    try:
        response = requests.get(f"{API_URL}/statistics")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_all_predictions():
    """Get all predictions from backend"""
    try:
        response = requests.get(f"{API_URL}/predictions")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# =============================================================================
# STEP 4: Check API Connection
# =============================================================================

if not check_api_connection():
    st.error("🔴 Backend API is not running!")
    st.info("Please start the backend API first: `python backend_api.py`")
    st.stop()
else:
    st.sidebar.success("✅ Connected to Backend API")

# =============================================================================
# STEP 5: Load Model Information
# =============================================================================

model_info = get_model_info()

# =============================================================================
# STEP 6: Main Application Header
# =============================================================================

st.title("🏦 Loan Approval Prediction System")
st.markdown("### 3-Tier Architecture: Frontend → Backend API → Database")
st.markdown("---")

# Display model information in sidebar
st.sidebar.header("📊 Model Information")
if model_info and model_info.get('model_trained'):
    accuracy = model_info.get('accuracy')
    if accuracy is not None:
        st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    else:
        st.sidebar.metric("Model Accuracy", "Training...")
    st.sidebar.markdown(
        f"**Training Records:** {model_info.get('total_records', 0)}")
else:
    st.sidebar.warning("Model not trained")

# Get and display statistics
stats = get_statistics()
if stats:
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Prediction Stats")
    st.sidebar.metric("Total Predictions", stats['total_predictions'])
    st.sidebar.metric("Approval Rate", f"{stats['approval_rate']:.1f}%")

# =============================================================================
# STEP 7: Create Tabs
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict Loan", "📊 Prediction History", "📈 Statistics", "ℹ️ About"])

# =============================================================================
# STEP 8: Prediction Tab
# =============================================================================

with tab1:
    st.header("Enter Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox(
            "Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox(
            "Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        st.subheader("Financial Information")
        applicant_income = st.number_input(
            "Applicant Income (₹)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)",
            min_value=0,
            max_value=100000,
            value=0,
            step=500
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ in thousands)",
            min_value=0,
            max_value=1000,
            value=150,
            step=10
        )
        loan_term = st.selectbox(
            "Loan Amount Term (months)",
            [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
        )
        credit_history = st.selectbox(
            "Credit History (1=Good, 0=Bad)",
            [1, 0]
        )

    st.markdown("---")

    if st.button("🔍 Predict Loan Approval", use_container_width=True):
        # Prepare application data
        application_data = {
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "property_area": property_area,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_history": credit_history
        }

        with st.spinner("🔄 Calling Backend API..."):
            result = make_prediction(application_data)

        if result:
            st.markdown("### Prediction Result")

            prediction = result['prediction']
            confidence = result['confidence']
            prob_approved = result['probability_approved']
            prob_rejected = result['probability_rejected']

            if prediction == 'Approved':
                # Celebration view
                st.balloons()
                st.markdown(
                    '<div class="prediction-box approved">✅ LOAN APPROVED</div>',
                    unsafe_allow_html=True
                )
                st.success(f"Confidence: {confidence:.2f}%")
                st.markdown(
                    "### 🎉 Congratulations! Your loan has been approved!")
            else:
                # Sad view with heartbreak
                st.markdown(
                    """
                    <style>
                    @keyframes heartbreak {
                        0% { transform: scale(1) rotate(0deg); opacity: 1; }
                        50% { transform: scale(1.2) rotate(10deg); opacity: 0.8; }
                        100% { transform: scale(0.8) rotate(-10deg); opacity: 0.6; }
                    }
                    .heartbreak-container {
                        text-align: center;
                        font-size: 80px;
                        animation: heartbreak 1s ease-in-out infinite;
                        margin: 20px 0;
                    }
                    </style>
                    <div class="heartbreak-container">💔💔💔</div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="prediction-box rejected">❌ LOAN REJECTED</div>',
                    unsafe_allow_html=True
                )
                st.error(f"Confidence: {confidence:.2f}%")
                st.markdown(
                    "### 😔 We're sorry. Your loan application was not approved at this time.")

            st.info(
                f"✓ Prediction saved to database (Record ID: {result['record_id']})")

            # Show probability chart
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_approved,
                    title={'text': "Approval Probability"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if prediction == 'Approved' else "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Income (Total)', 'Loan Amount', 'Loan Term', 'Credit History', 'Education', 'Property Area'],
                    'Value': [
                        f"₹{applicant_income + coapplicant_income:,}",
                        f"₹{loan_amount:,}k",
                        f"{loan_term} months",
                        "Good" if credit_history == 1 else "Bad",
                        education,
                        property_area
                    ]
                })
                st.dataframe(summary_df, use_container_width=True,
                             hide_index=True)

# =============================================================================
# STEP 9: Prediction History Tab
# =============================================================================

with tab2:
    st.header("📊 Prediction History")

    predictions_data = get_all_predictions()

    if predictions_data and predictions_data['total'] > 0:
        predictions_list = predictions_data['predictions']
        df_predictions = pd.DataFrame(predictions_list)

        st.metric("Total Records", predictions_data['total'])

        # Display as dataframe
        st.dataframe(
            df_predictions[['id', 'timestamp', 'prediction', 'confidence',
                            'applicant_income', 'loan_amount', 'education', 'property_area']],
            use_container_width=True
        )

        # Download button
        csv = df_predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="loan_predictions_history.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "No predictions yet. Make your first prediction in the 'Predict Loan' tab!")

# =============================================================================
# STEP 10: Statistics Tab
# =============================================================================

with tab3:
    st.header("📈 Prediction Statistics")

    if stats and stats['total_predictions'] > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        with col2:
            st.metric("Approved", stats['total_approved'])
        with col3:
            st.metric("Rejected", stats['total_rejected'])
        with col4:
            st.metric("Approval Rate", f"{stats['approval_rate']:.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig = px.pie(
                values=[stats['total_approved'], stats['total_rejected']],
                names=['Approved', 'Rejected'],
                title="Approval vs Rejection",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar chart
            fig = px.bar(
                x=['Approved', 'Rejected'],
                y=[stats['total_approved'], stats['total_rejected']],
                title="Prediction Count",
                labels={'x': 'Prediction', 'y': 'Count'},
                color=['Approved', 'Rejected'],
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Average Confidence", f"{stats['average_confidence']:.2f}%")
    else:
        st.info("No statistics available yet. Make some predictions first!")

# =============================================================================
# STEP 11: About Tab
# =============================================================================

with tab4:
    st.header("ℹ️ About This Application")

    st.markdown("""
    ### 🏗️ 3-Tier Architecture
    
    This application demonstrates a modern **3-tier architecture**:
    
    #### 1. 🎨 **Frontend Layer** (Streamlit)
    - User interface for input and visualization
    - Communicates with backend via REST API
    - File: `ML_Streamlit_Loan_API.py`
    
    #### 2. ⚙️ **Backend Layer** (FastAPI)
    - REST API endpoints for predictions
    - Machine learning model training and inference
    - Business logic layer
    - File: `backend_api.py`
    - API Docs: http://localhost:8000/docs
    
    #### 3. 💾 **Database Layer** (SQLite)
    - Data persistence and retrieval
    - Stores all predictions with timestamps
    - File: `db_layer.py`
    - Database: `loan_predictions.db`
    
    ---
    
    ### 🎯 Features
    
    ✅ **Real-time Predictions** - Get instant loan approval decisions  
    ✅ **Visual Feedback** - Balloons for approvals, heartbreak for rejections  
    ✅ **Data Persistence** - All predictions saved to database  
    ✅ **Statistics Dashboard** - Track approval rates and trends  
    ✅ **REST API** - Fully documented API endpoints  
    ✅ **Scalable Design** - Each layer can be scaled independently  
    
    ---
    
    ### 🤖 Machine Learning Model
    
    - **Algorithm:** Logistic Regression
    - **Features:** Income, loan amount, credit history, education, property area, etc.
    - **Preprocessing:** Log transformation, scaling, encoding
    - **Accuracy:** {:.2f}%
    
    ---
    
    ### 🚀 How to Run
    
    1. **Start Backend API:**
       ```bash
       python backend_api.py
       ```
       API will run on http://localhost:8000
    
    2. **Start Frontend:**
       ```bash
       streamlit run ML_Streamlit_Loan_API.py
       ```
       Frontend will run on http://localhost:8501
    
    ---
    
    ### 📚 API Endpoints
    
    - `GET /` - API information
    - `POST /train` - Train the model
    - `GET /model-info` - Get model information
    - `POST /predict` - Make a prediction
    - `GET /predictions` - Get all predictions
    - `GET /statistics` - Get prediction statistics
    
    View full API documentation at: http://localhost:8000/docs
    
    ---
    
    **Created with ❤️ using Streamlit, FastAPI, and scikit-learn**
    """.format((model_info.get('accuracy', 0) or 0) * 100 if model_info else 0))

# =============================================================================
# STEP 12: Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Ensure Backend API is running
- Higher income increases approval
- Good credit history is crucial
- Graduate education helps
- All predictions are saved to database
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.markdown("[API Documentation](http://localhost:8000/docs)")
st.sidebar.markdown("[API Root](http://localhost:8000)")
