# =============================================================================
# STEP 1: Import Required Libraries
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime

# =============================================================================
# STEP 2: Configure Streamlit Page
# =============================================================================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STEP 3: Add Custom CSS for Better Styling
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
# STEP 4: Initialize SQLite Database
# =============================================================================


def init_database():
    """Initialize SQLite database and create predictions table"""
    conn = sqlite3.connect('loan_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            married TEXT,
            dependents TEXT,
            education TEXT,
            self_employed TEXT,
            property_area TEXT,
            applicant_income REAL,
            coapplicant_income REAL,
            loan_amount REAL,
            loan_term INTEGER,
            credit_history INTEGER,
            prediction TEXT,
            confidence REAL
        )
    ''')

    conn.commit()
    conn.close()


def save_prediction(input_data, prediction, confidence):
    """Save prediction to SQLite database"""
    conn = sqlite3.connect('loan_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions (
            timestamp, married, dependents, education, self_employed,
            property_area, applicant_income, coapplicant_income,
            loan_amount, loan_term, credit_history, prediction, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        input_data['Married'].iloc[0],
        input_data['Dependents'].iloc[0],
        input_data['Education'].iloc[0],
        input_data['Self_Employed'].iloc[0],
        input_data['Property_Area'].iloc[0],
        input_data['ApplicantIncome'].iloc[0],
        input_data['CoapplicantIncome'].iloc[0],
        input_data['LoanAmount'].iloc[0],
        input_data['Loan_Amount_Term'].iloc[0],
        input_data['Credit_History'].iloc[0],
        'Approved' if prediction == 1 else 'Rejected',
        confidence
    ))

    conn.commit()
    conn.close()

# =============================================================================
# STEP 5: Load and Prepare Data
# =============================================================================


@st.cache_data
def load_data():
    """Load the loan dataset"""
    df = pd.read_csv("Loan_dataset.csv")
    df = df.dropna(subset=["Loan_Status"])
    return df

# =============================================================================
# STEP 6: Train the Machine Learning Model
# =============================================================================


@st.cache_resource
def train_model():
    """Train the logistic regression model with pipeline"""

    # Load data
    df = load_data()

    # Prepare features and target
    X = df.drop(columns=["Loan_Status", "Loan_ID", "Gender"])
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define column types
    log_cols = ["ApplicantIncome", "CoapplicantIncome"]
    num_cols = ["LoanAmount", "Credit_History", "Loan_Amount_Term"]
    cat_cols = ["Married", "Self_Employed",
                "Education", "Property_Area", "Dependents"]

    # Create pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    log_numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("log_num", log_numeric_pipeline, log_cols),
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # Create and train model pipeline
    model_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    model_pipeline.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model_pipeline, accuracy, X_test, y_test


# =============================================================================
# STEP 7: Create Main Application Header
# =============================================================================
# Initialize database
init_database()

st.title("🏬 Loan Approval Prediction System")
st.markdown("### Predict whether a loan application will be approved or rejected")
st.markdown("---")

# =============================================================================
# STEP 8: Train Model and Display Model Information
# =============================================================================

# =============================================================================
# STEP 7: Train Model and Display Model Information
# =============================================================================
with st.spinner("Loading model..."):
    model, accuracy, X_test, y_test = train_model()
    df = load_data()

# Display model accuracy in sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.markdown(f"**Total Records:** {len(df)}")
st.sidebar.markdown(f"**Approved Loans:** {(df['Loan_Status'] == 'Y').sum()}")
st.sidebar.markdown(f"**Rejected Loans:** {(df['Loan_Status'] == 'N').sum()}")

# =============================================================================
# STEP 9: Create Tabs for Different Sections
# =============================================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict Loan", "📈 Data Analysis", "ℹ️ About"])

# =============================================================================
# STEP 9: Prediction Tab - Create Input Form
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

    # =============================================================================
    # STEP 10: Make Prediction
    # =============================================================================
    st.markdown("---")
    if st.button("🔍 Predict Loan Approval", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Married': [married],
            'Self_Employed': [self_employed],
            'Education': [education],
            'Property_Area': [property_area],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Credit_History': [credit_history],
            'Loan_Amount_Term': [loan_term],
            'Dependents': [dependents]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display result
        st.markdown("### Prediction Result")

        if prediction == 1:
            # Celebration view for approved loan
            st.balloons()
            st.markdown(
                f'<div class="prediction-box approved">✅ LOAN APPROVED</div>',
                unsafe_allow_html=True
            )
            st.success(f"Confidence: {prediction_proba[1]*100:.2f}%")
            st.markdown("### 🎉 Congratulations! Your loan has been approved!")
        else:
            # Sad view for rejected loan with heartbreak
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
                f'<div class="prediction-box rejected">❌ LOAN REJECTED</div>',
                unsafe_allow_html=True
            )
            st.error(f"Confidence: {prediction_proba[0]*100:.2f}%")
            st.markdown(
                "### 😔 We're sorry. Your loan application was not approved at this time.")

        # Save prediction to database
        save_prediction(input_data, prediction,
                        prediction_proba[prediction]*100)
        st.info("✓ Prediction saved to database")

        # Show probability chart
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba[1]*100,
                title={'text': "Approval Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
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
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# =============================================================================
# STEP 11: Data Analysis Tab
# =============================================================================
with tab2:
    st.header("📊 Loan Dataset Analysis")

    # Display dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Applications", len(df))
    with col2:
        approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    with col3:
        avg_income = df['ApplicantIncome'].mean()
        st.metric("Avg Income", f"₹{avg_income:,.0f}")
    with col4:
        avg_loan = df['LoanAmount'].mean()
        st.metric("Avg Loan", f"₹{avg_loan:.0f}k")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Loan status distribution
        status_counts = df['Loan_Status'].value_counts()
        fig1 = px.pie(
            values=status_counts.values,
            names=['Approved', 'Rejected'],
            title="Loan Approval Status Distribution",
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Approval by property area
        approval_by_area = df.groupby('Property_Area')['Loan_Status'].apply(
            lambda x: (x == 'Y').sum()
        ).reset_index()
        approval_by_area.columns = ['Property_Area', 'Approved']

        fig2 = px.bar(
            approval_by_area,
            x='Property_Area',
            y='Approved',
            title="Loan Approvals by Property Area",
            color='Approved',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Income distribution
        fig3 = px.histogram(
            df,
            x='ApplicantIncome',
            nbins=30,
            title="Applicant Income Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig3.update_xaxes(title="Income (₹)")
        fig3.update_yaxes(title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Loan amount distribution
        fig4 = px.histogram(
            df.dropna(subset=['LoanAmount']),
            x='LoanAmount',
            nbins=30,
            title="Loan Amount Distribution",
            color_discrete_sequence=['#9b59b6']
        )
        fig4.update_xaxes(title="Loan Amount (₹ in thousands)")
        fig4.update_yaxes(title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# =============================================================================
# STEP 12: About Tab
# =============================================================================
with tab3:
    st.header("ℹ️ About This Application")

    st.markdown("""
    ### Loan Approval Prediction System
    
    This application uses **Machine Learning** to predict whether a loan application 
    will be approved or rejected based on various applicant features.
    
    #### 🎯 Features Used for Prediction:
    - **Personal Information:** Marital status, dependents, education, employment
    - **Financial Information:** Income, loan amount, credit history
    - **Property Information:** Property area (Urban/Semiurban/Rural)
    
    #### 🤖 Machine Learning Model:
    - **Algorithm:** Logistic Regression with balanced class weights
    - **Preprocessing:** 
        - Log transformation for income features
        - Standard scaling for numerical features
        - One-hot encoding for categorical features
        - Missing value imputation
    - **Accuracy:** {:.2f}%
    
    #### 📊 Dataset Information:
    - **Source:** Loan Dataset (CSV)
    - **Total Records:** {}
    - **Features:** 11 input features + 1 target variable
    
    #### 💡 How to Use:
    1. Go to the **Predict Loan** tab
    2. Enter the applicant's details in the form
    3. Click on **Predict Loan Approval** button
    4. View the prediction result and confidence score
    
    #### 🛠️ Technologies Used:
    - **Streamlit** - Web application framework
    - **Scikit-learn** - Machine learning library
    - **Pandas** - Data manipulation
    - **Plotly** - Interactive visualizations
    
    ---
    
    **Note:** This is a demonstration application. Actual loan approval decisions 
    should involve comprehensive financial analysis and human oversight.
    """.format(accuracy*100, len(df)))

    st.markdown("---")
    st.markdown("**Created with ❤️ using Streamlit and Python**")

# =============================================================================
# STEP 13: Footer
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("""
- Higher income increases approval chances
- Good credit history is crucial
- Lower loan amounts are easier to approve
- Graduate education helps
- Urban properties have higher approval rates
""")
