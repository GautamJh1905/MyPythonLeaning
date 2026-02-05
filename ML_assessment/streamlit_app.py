"""
Customer Segmentation Dashboard - Streamlit Application
Interactive web interface for customer segment prediction and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .segment-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data


@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        with open('models/kmeans_customer_segmentation.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encodings.pkl', 'rb') as f:
            encodings = pickle.load(f)
        return kmeans, scaler, encodings, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, False


@st.cache_data
def load_customer_data():
    """Load customer data for analysis"""
    try:
        df = pd.read_csv('CustomerData_clean.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Segment definitions
SEGMENT_INFO = {
    0: {
        'name': 'Price-Sensitive Occasional',
        'color': '#FF6B6B',
        'icon': '💰',
        'description': 'Low-income, low-spending customers with minimal engagement',
        'characteristics': [
            'Low annual income (~₹262K)',
            'Minimal spending (~₹54K)',
            'Infrequent purchases (~2.2/month)',
            'Low app engagement (~18 min)'
        ],
        'recommendations': [
            'Aggressive discounts and promotional codes',
            'Free shipping on first order',
            'Entry-level loyalty programs',
            'Budget-friendly product recommendations'
        ]
    },
    1: {
        'name': 'High-Value Loyal',
        'color': '#4ECDC4',
        'icon': '👑',
        'description': 'Premium customers with highest income and engagement',
        'characteristics': [
            'High annual income (~₹1.03M)',
            'High spending (~₹663K)',
            'Very frequent purchases (~16.1/month)',
            'High app engagement (~116 min)'
        ],
        'recommendations': [
            'Exclusive VIP member benefits',
            'Early access to new products',
            'Premium customer service',
            'Personalized luxury recommendations'
        ]
    },
    2: {
        'name': 'Value-Seeking Regular',
        'color': '#45B7D1',
        'icon': '⭐',
        'description': 'Mid-tier customers with moderate income and regular engagement',
        'characteristics': [
            'Mid-range income (~₹585K)',
            'Moderate spending (~₹260K)',
            'Regular purchases (~8.8/month)',
            'Moderate engagement (~68 min)'
        ],
        'recommendations': [
            'Seasonal sales and limited-time offers',
            'Bundle deals and combo packages',
            'Referral bonuses',
            'Mid-range product recommendations'
        ]
    }
}


def predict_segment(customer_data, kmeans, scaler, encodings):
    """Predict customer segment"""
    try:
        # Create DataFrame
        df = pd.DataFrame([customer_data])

        # Encode categorical variables
        for col, encoding in encodings.items():
            if col in df.columns:
                df[col] = df[col].map(encoding['mapping'])

        # Scale features
        scaled_data = scaler.transform(df)

        # Predict cluster
        cluster = kmeans.predict(scaled_data)[0]

        # Calculate confidence (distance to cluster centers)
        distances = kmeans.transform(scaled_data)[0]
        total_distance = sum(distances)
        confidence = (1 - distances[cluster] / total_distance) * 100

        return cluster, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Segmentation Dashboard</h1>',
                unsafe_allow_html=True)

    # Load models
    kmeans, scaler, encodings, models_loaded = load_models()
    df = load_customer_data()

    if not models_loaded or df is None:
        st.error(
            "⚠️ Failed to load models or data. Please check if model files exist.")
        return

    # Sidebar
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/customer-insight.png", width=100)
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔮 Predict Segment", "📊 Analytics Dashboard",
                "👥 Customer Analysis", "📈 Business Insights"],
            label_visibility="collapsed"
        )

        st.divider()

        # Quick stats
        if df is not None:
            st.metric("Total Customers", len(df))
            st.metric("Total Segments", 3)
            st.metric("Total Revenue",
                      f"₹{df['TotalSpent'].sum()/1_000_000:.2f}M")

    # Main content
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "🔮 Predict Segment":
        show_prediction_page(kmeans, scaler, encodings)
    elif page == "📊 Analytics Dashboard":
        show_analytics_page(df)
    elif page == "👥 Customer Analysis":
        show_customer_analysis_page(df)
    elif page == "📈 Business Insights":
        show_business_insights_page(df)


def show_home_page(df):
    """Home page with overview"""
    st.header("Welcome to Customer Segmentation Dashboard")

    st.markdown("""
    This intelligent dashboard helps you understand your customer base through advanced machine learning segmentation.
    
    ### 🎯 Key Features:
    - **Predict Customer Segments**: Input customer details and get instant segment predictions
    - **Analytics Dashboard**: Visualize customer distribution and segment characteristics
    - **Customer Analysis**: Deep dive into demographics and behavior patterns
    - **Business Insights**: Actionable recommendations for each segment
    """)

    # Quick overview metrics
    st.subheader("📊 Quick Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(df), help="Total customers analyzed")

    with col2:
        avg_value = df['TotalSpent'].mean()
        st.metric("Avg Customer Value",
                  f"₹{avg_value/1000:.0f}K", help="Average spending per customer")

    with col3:
        total_revenue = df['TotalSpent'].sum()
        st.metric("Total Revenue",
                  f"₹{total_revenue/1_000_000:.2f}M", help="Total revenue generated")

    with col4:
        avg_purchases = df['MonthlyPurchases'].mean()
        st.metric("Avg Purchases/Month",
                  f"{avg_purchases:.1f}", help="Average monthly purchases")

    st.divider()

    # Segment overview
    st.subheader("🎯 Customer Segments Overview")

    # Add Cluster column if not present
    if 'Cluster' not in df.columns:
        # Load the model to predict clusters
        kmeans, scaler, encodings, _ = load_models()
        if kmeans and scaler:
            df_temp = df.copy()
            # Encode and scale
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = df_temp.select_dtypes(
                include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in encodings:
                    le = LabelEncoder()
                    le.classes_ = np.array(encodings[col]['classes'])
                    df_temp[col] = le.transform(df_temp[col])

            df_temp = df_temp.drop('Cluster', axis=1, errors='ignore')
            scaled = scaler.transform(df_temp)
            df['Cluster'] = kmeans.predict(scaled)

    col1, col2, col3 = st.columns(3)

    for idx, (col, cluster_id) in enumerate(zip([col1, col2, col3], [0, 1, 2])):
        with col:
            info = SEGMENT_INFO[cluster_id]
            count = len(df[df['Cluster'] == cluster_id])
            percentage = (count / len(df)) * 100

            st.markdown(f"""
            <div style="background-color: {info['color']}22; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {info['color']}">
                <h3 style="color: {info['color']};">{info['icon']} {info['name']}</h3>
                <p style="font-size: 0.9rem; color: #666;">{info['description']}</p>
                <h2 style="color: {info['color']};">{count} customers ({percentage:.1f}%)</h2>
            </div>
            """, unsafe_allow_html=True)


def show_prediction_page(kmeans, scaler, encodings):
    """Customer segment prediction page"""
    st.header("🔮 Predict Customer Segment")
    st.markdown(
        "Enter customer details to predict their segment and receive personalized recommendations.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Information")
        customer_id = st.text_input(
            "Customer ID", "C001", help="Unique customer identifier")
        age = st.number_input("Age", min_value=18, max_value=100,
                              value=35, help="Customer age in years")
        gender = st.selectbox("Gender", ["M", "F"], help="Customer gender")
        city = st.selectbox("City", ["Bangalore", "Mumbai", "Delhi", "Pune", "Chennai", "Kolkata", "Hyderabad"],
                            help="Customer city")

    with col2:
        st.subheader("Financial & Behavioral Data")
        annual_income = st.number_input("Annual Income (₹)", min_value=0, max_value=5000000, value=600000,
                                        step=10000, help="Annual income in rupees")
        total_spent = st.number_input("Total Spent (₹)", min_value=0, max_value=2000000, value=250000,
                                      step=10000, help="Total amount spent")
        monthly_purchases = st.number_input("Monthly Purchases", min_value=0, max_value=30, value=9,
                                            help="Average purchases per month")
        avg_order_value = st.number_input("Avg Order Value (₹)", min_value=0, max_value=50000, value=3500,
                                          step=100, help="Average order value")

    col3, col4 = st.columns(2)

    with col3:
        app_time = st.number_input("App Time (minutes)", min_value=0, max_value=200, value=70,
                                   help="Average app usage time in minutes")

    with col4:
        discount_usage = st.selectbox("Discount Usage", ["Low", "Medium", "High"],
                                      help="Frequency of discount usage")
        shopping_time = st.selectbox("Preferred Shopping Time", ["Day", "Night"],
                                     help="Preferred shopping time")

    st.divider()

    if st.button("🎯 Predict Segment", type="primary", use_container_width=True):
        # Prepare customer data
        customer_data = {
            'CustomerID': customer_id,
            'Age': age,
            'Gender': gender,
            'City': city,
            'AnnualIncome': annual_income,
            'TotalSpent': total_spent,
            'MonthlyPurchases': monthly_purchases,
            'AvgOrderValue': avg_order_value,
            'AppTimeMinutes': app_time,
            'DiscountUsage': discount_usage,
            'PreferredShoppingTime': shopping_time
        }

        # Predict
        cluster, confidence = predict_segment(
            customer_data, kmeans, scaler, encodings)

        if cluster is not None:
            info = SEGMENT_INFO[cluster]

            # Display results
            st.success("✅ Prediction Successful!")

            # Main result card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%); 
                        padding: 2rem; border-radius: 1rem; border: 2px solid {info['color']}; margin: 1rem 0;">
                <h2 style="color: {info['color']};">{info['icon']} {info['name']}</h2>
                <p style="font-size: 1.1rem;">{info['description']}</p>
                <h3 style="color: {info['color']};">Confidence: {confidence:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Segment Characteristics")
                for char in info['characteristics']:
                    st.markdown(f"- {char}")

            with col2:
                st.subheader("💡 Recommended Actions")
                for rec in info['recommendations']:
                    st.markdown(f"- {rec}")

            # Customer profile comparison
            st.subheader("📊 Customer Profile")
            profile_col1, profile_col2, profile_col3 = st.columns(3)

            with profile_col1:
                income_level = "High" if annual_income > 800000 else "Medium" if annual_income > 400000 else "Low"
                st.metric("Income Level", income_level)

            with profile_col2:
                spending_pattern = "High" if total_spent > 500000 else "Medium" if total_spent > 200000 else "Low"
                st.metric("Spending Pattern", spending_pattern)

            with profile_col3:
                engagement = "High" if app_time > 90 else "Medium" if app_time > 50 else "Low"
                st.metric("Engagement Level", engagement)


def show_analytics_page(df):
    """Analytics dashboard with visualizations"""
    st.header("📊 Analytics Dashboard")

    # Ensure Cluster column exists
    if 'Cluster' not in df.columns:
        kmeans, scaler, encodings, _ = load_models()
        if kmeans and scaler:
            df_temp = df.copy()
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = df_temp.select_dtypes(
                include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in encodings:
                    le = LabelEncoder()
                    le.classes_ = np.array(encodings[col]['classes'])
                    df_temp[col] = le.transform(df_temp[col])

            df_temp = df_temp.drop('Cluster', axis=1, errors='ignore')
            scaled = scaler.transform(df_temp)
            df['Cluster'] = kmeans.predict(scaled)

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(
        ["📈 Distribution", "💰 Revenue", "👥 Demographics"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Cluster distribution pie chart
            cluster_counts = df['Cluster'].value_counts().sort_index()
            labels = [SEGMENT_INFO[i]['name'] for i in cluster_counts.index]
            colors = [SEGMENT_INFO[i]['color'] for i in cluster_counts.index]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=cluster_counts.values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(size=12)
            )])
            fig.update_layout(
                title="Customer Segment Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar chart
            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=cluster_counts.values,
                marker_color=colors,
                text=cluster_counts.values,
                textposition='outside'
            )])
            fig.update_layout(
                title="Customer Count by Segment",
                xaxis_title="Segment",
                yaxis_title="Number of Customers",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Revenue analysis
        revenue_by_cluster = df.groupby('Cluster')['TotalSpent'].sum()
        avg_revenue = df.groupby('Cluster')['TotalSpent'].mean()

        col1, col2 = st.columns(2)

        with col1:
            labels = [SEGMENT_INFO[i]['name']
                      for i in revenue_by_cluster.index]
            colors = [SEGMENT_INFO[i]['color']
                      for i in revenue_by_cluster.index]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=revenue_by_cluster.values / 1000,
                marker_color=colors,
                text=[f'₹{v/1000:.0f}K' for v in revenue_by_cluster.values],
                textposition='outside'
            )])
            fig.update_layout(
                title="Total Revenue by Segment",
                xaxis_title="Segment",
                yaxis_title="Revenue (₹ Thousands)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=avg_revenue.values / 1000,
                marker_color=colors,
                text=[f'₹{v/1000:.0f}K' for v in avg_revenue.values],
                textposition='outside'
            )])
            fig.update_layout(
                title="Average Revenue per Customer",
                xaxis_title="Segment",
                yaxis_title="Avg Revenue (₹ Thousands)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Revenue metrics
        st.subheader("💰 Revenue Metrics")
        col1, col2, col3 = st.columns(3)

        total_revenue = df['TotalSpent'].sum()
        with col1:
            st.metric("Total Revenue", f"₹{total_revenue/1_000_000:.2f}M")

        with col2:
            max_cluster = revenue_by_cluster.idxmax()
            st.metric("Highest Revenue Segment",
                      SEGMENT_INFO[max_cluster]['name'])

        with col3:
            max_clv_cluster = avg_revenue.idxmax()
            st.metric("Most Valuable Segment",
                      SEGMENT_INFO[max_clv_cluster]['name'])

    with tab3:
        # Demographics analysis
        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            fig = go.Figure()
            for cluster in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster]
                fig.add_trace(go.Histogram(
                    x=cluster_data['Age'],
                    name=SEGMENT_INFO[cluster]['name'],
                    marker_color=SEGMENT_INFO[cluster]['color'],
                    opacity=0.7
                ))
            fig.update_layout(
                title="Age Distribution by Segment",
                xaxis_title="Age",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # App engagement
            app_time = df.groupby('Cluster')['AppTimeMinutes'].mean()
            labels = [SEGMENT_INFO[i]['name'] for i in app_time.index]
            colors = [SEGMENT_INFO[i]['color'] for i in app_time.index]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=app_time.values,
                marker_color=colors,
                text=[f'{v:.1f}m' for v in app_time.values],
                textposition='outside'
            )])
            fig.update_layout(
                title="App Engagement by Segment",
                xaxis_title="Segment",
                yaxis_title="Average App Time (minutes)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def show_customer_analysis_page(df):
    """Detailed customer analysis"""
    st.header("👥 Customer Analysis")

    # Ensure Cluster column exists
    if 'Cluster' not in df.columns:
        kmeans, scaler, encodings, _ = load_models()
        if kmeans and scaler:
            df_temp = df.copy()
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = df_temp.select_dtypes(
                include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in encodings:
                    le = LabelEncoder()
                    le.classes_ = np.array(encodings[col]['classes'])
                    df_temp[col] = le.transform(df_temp[col])

            df_temp = df_temp.drop('Cluster', axis=1, errors='ignore')
            scaled = scaler.transform(df_temp)
            df['Cluster'] = kmeans.predict(scaled)

    # Segment selector
    segment = st.selectbox(
        "Select Segment to Analyze",
        options=[0, 1, 2],
        format_func=lambda x: f"{SEGMENT_INFO[x]['icon']} {SEGMENT_INFO[x]['name']}"
    )

    cluster_data = df[df['Cluster'] == segment]
    info = SEGMENT_INFO[segment]

    # Segment overview
    st.markdown(f"""
    <div style="background-color: {info['color']}22; padding: 1.5rem; border-radius: 0.5rem; 
                border-left: 4px solid {info['color']}; margin: 1rem 0;">
        <h2 style="color: {info['color']};">{info['icon']} {info['name']}</h2>
        <p>{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", len(cluster_data))

    with col2:
        avg_income = cluster_data['AnnualIncome'].mean()
        st.metric("Avg Annual Income", f"₹{avg_income/1000:.0f}K")

    with col3:
        avg_spent = cluster_data['TotalSpent'].mean()
        st.metric("Avg Total Spent", f"₹{avg_spent/1000:.0f}K")

    with col4:
        avg_purchases = cluster_data['MonthlyPurchases'].mean()
        st.metric("Avg Monthly Purchases", f"{avg_purchases:.1f}")

    st.divider()

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Demographics")

        # Age stats
        st.write(
            f"**Age Range:** {cluster_data['Age'].min()}-{cluster_data['Age'].max()} years")
        st.write(f"**Average Age:** {cluster_data['Age'].mean():.1f} years")

        # Gender distribution
        gender_dist = cluster_data['Gender'].value_counts()
        st.write("**Gender Distribution:**")
        for gender, count in gender_dist.items():
            pct = (count / len(cluster_data)) * 100
            st.write(f"- {gender}: {count} ({pct:.1f}%)")

    with col2:
        st.subheader("💳 Spending Behavior")

        st.write(
            f"**Avg Order Value:** ₹{cluster_data['AvgOrderValue'].mean():,.0f}")
        st.write(
            f"**Purchase Frequency:** {cluster_data['MonthlyPurchases'].mean():.1f}/month")
        st.write(
            f"**App Engagement:** {cluster_data['AppTimeMinutes'].mean():.1f} min/session")

        # Discount usage
        discount_dist = cluster_data['DiscountUsage'].value_counts()
        st.write("**Discount Usage:**")
        for usage, count in discount_dist.items():
            pct = (count / len(cluster_data)) * 100
            st.write(f"- {usage}: {count} ({pct:.1f}%)")

    # Sample customers
    st.subheader("📋 Sample Customers")
    sample_cols = ['CustomerID', 'Age', 'Gender', 'City',
                   'AnnualIncome', 'TotalSpent', 'MonthlyPurchases']
    st.dataframe(cluster_data[sample_cols].head(10), use_container_width=True)


def show_business_insights_page(df):
    """Business insights and recommendations"""
    st.header("📈 Business Insights & Recommendations")

    # Ensure Cluster column exists
    if 'Cluster' not in df.columns:
        kmeans, scaler, encodings, _ = load_models()
        if kmeans and scaler:
            df_temp = df.copy()
            from sklearn.preprocessing import LabelEncoder
            categorical_cols = df_temp.select_dtypes(
                include=['object']).columns.tolist()
            for col in categorical_cols:
                if col in encodings:
                    le = LabelEncoder()
                    le.classes_ = np.array(encodings[col]['classes'])
                    df_temp[col] = le.transform(df_temp[col])

            df_temp = df_temp.drop('Cluster', axis=1, errors='ignore')
            scaled = scaler.transform(df_temp)
            df['Cluster'] = kmeans.predict(scaled)

    # Key metrics
    st.subheader("🎯 Key Business Metrics")

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)
    total_revenue = df['TotalSpent'].sum()
    avg_clv = total_revenue / total_customers

    with col1:
        st.metric("Total Customer Base", total_customers)

    with col2:
        st.metric("Total Revenue", f"₹{total_revenue/1_000_000:.2f}M")

    with col3:
        st.metric("Avg Customer Value", f"₹{avg_clv/1000:.0f}K")

    with col4:
        revenue_by_cluster = df.groupby('Cluster')['TotalSpent'].sum()
        top_segment = SEGMENT_INFO[revenue_by_cluster.idxmax()]['name']
        st.metric("Top Revenue Segment", top_segment)

    st.divider()

    # Strategic recommendations
    st.subheader("🎯 Strategic Recommendations by Segment")

    for cluster_id in [0, 1, 2]:
        info = SEGMENT_INFO[cluster_id]
        cluster_data = df[df['Cluster'] == cluster_id]

        with st.expander(f"{info['icon']} {info['name']}", expanded=(cluster_id == 1)):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**📊 Segment Profile:**")
                st.write(
                    f"- Size: {len(cluster_data)} customers ({len(cluster_data)/total_customers*100:.1f}%)")
                st.write(
                    f"- Revenue: ₹{revenue_by_cluster[cluster_id]/1_000_000:.2f}M ({revenue_by_cluster[cluster_id]/total_revenue*100:.1f}%)")
                st.write(
                    f"- CLV: ₹{revenue_by_cluster[cluster_id]/len(cluster_data)/1000:.0f}K")

                st.markdown("**💡 Key Characteristics:**")
                for char in info['characteristics']:
                    st.write(f"- {char}")

            with col2:
                st.markdown("**🎯 Recommended Actions:**")
                for idx, rec in enumerate(info['recommendations'], 1):
                    st.write(f"{idx}. {rec}")

                st.markdown("**💼 Business Impact:**")
                if cluster_id == 0:
                    st.write("- Focus on conversion over revenue")
                    st.write("- Reduce acquisition costs")
                    st.write("- Create gateway products")
                elif cluster_id == 1:
                    st.write("- Maximize retention (highest value)")
                    st.write("- Invest in premium service")
                    st.write("- Monitor churn risk closely")
                else:
                    st.write("- Increase purchase frequency")
                    st.write("- Create loyalty programs")
                    st.write("- Design upgrade path to premium")

    st.divider()

    # Growth opportunities
    st.subheader("📈 Growth Opportunities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Revenue Optimization:**
        - High-Value Loyal generates highest per-customer revenue
        - Focus retention programs on Cluster 1
        - Implement win-back campaigns for at-risk customers
        
        **2. Customer Base Growth:**
        - Value-Seeking Regular is largest segment
        - Opportunity to move Cluster 2 → Cluster 1
        - Design targeted upgrade campaigns
        """)

    with col2:
        st.markdown("""
        **3. Cost Efficiency:**
        - Price-Sensitive has lowest CLV
        - Optimize acquisition costs for Cluster 0
        - Use automated marketing for efficiency
        
        **4. CLV Improvement:**
        - Target: 20% increase through segmented marketing
        - Strategy: Move customers up value chain
        - Monitor and act on migration patterns
        """)


if __name__ == "__main__":
    main()
