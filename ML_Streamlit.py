import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Temperature Data Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #31333F !important;
    }
    .stMetric .metric-value {
        color: #0e1117 !important;
    }
    [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h3 {
        color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🌡️ Interactive Temperature Data Dashboard")
st.markdown(
    "### Explore weather patterns and trends with interactive visualizations")

# Load data


@st.cache_data
def load_data():
    df = pd.read_csv('Temprature_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_year'] = df['date'].dt.dayofyear
    return df


try:
    df = load_data()

    # Sidebar filters
    st.sidebar.header("🔍 Filter Data")

    # Year filter
    years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        options=years,
        default=years
    )

    # Month filter
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    selected_months = st.sidebar.multiselect(
        "Select Month(s)",
        options=months,
        default=months
    )

    # Temperature range filter
    temp_range = st.sidebar.slider(
        "Temperature Range (°C)",
        float(df['temperature_c'].min()),
        float(df['temperature_c'].max()),
        (float(df['temperature_c'].min()), float(df['temperature_c'].max()))
    )

    # Filter data
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['month_name'].isin(selected_months)) &
        (df['temperature_c'] >= temp_range[0]) &
        (df['temperature_c'] <= temp_range[1])
    ]

    # Display metrics
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("🌡️ Avg Temperature",
                  f"{filtered_df['temperature_c'].mean():.2f}°C")
    with col3:
        st.metric("💧 Avg Humidity",
                  f"{filtered_df['humidity_percent'].mean():.2f}%")
    with col4:
        st.metric("🌧️ Total Rainfall",
                  f"{filtered_df['rainfall_mm'].sum():.2f}mm")
    with col5:
        st.metric("☀️ Avg Sunshine",
                  f"{filtered_df['sunshine_hours'].mean():.2f}hrs")

    st.markdown("---")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Temperature Trends", "🌦️ Weather Patterns", "📊 Correlations", "📅 Seasonal Analysis"])

    with tab1:
        st.subheader("Temperature Over Time")

        # Time series plot
        fig = px.line(
            filtered_df,
            x='date',
            y='temperature_c',
            title='Temperature Trend',
            labels={'temperature_c': 'Temperature (°C)', 'date': 'Date'}
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Temperature distribution
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                filtered_df,
                x='temperature_c',
                nbins=50,
                title='Temperature Distribution',
                labels={'temperature_c': 'Temperature (°C)'}
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(
                filtered_df,
                y='temperature_c',
                title='Temperature Box Plot',
                labels={'temperature_c': 'Temperature (°C)'}
            )
            fig_box.update_layout(height=350)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.subheader("Weather Patterns Analysis")

        # Multi-variable time series
        weather_vars = st.multiselect(
            "Select weather variables to visualize:",
            ['temperature_c', 'humidity_percent', 'pressure_hpa', 'wind_speed_kmph',
             'cloud_cover_percent', 'rainfall_mm', 'sunshine_hours'],
            default=['temperature_c', 'humidity_percent', 'rainfall_mm']
        )

        if weather_vars:
            fig_multi = go.Figure()
            for var in weather_vars:
                fig_multi.add_trace(go.Scatter(
                    x=filtered_df['date'],
                    y=filtered_df[var],
                    name=var.replace('_', ' ').title(),
                    mode='lines'
                ))
            fig_multi.update_layout(
                title='Multiple Weather Variables Over Time',
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_multi, use_container_width=True)

        # Scatter plots
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter1 = px.scatter(
                filtered_df,
                x='humidity_percent',
                y='temperature_c',
                color='rainfall_mm',
                title='Temperature vs Humidity (colored by Rainfall)',
                labels={
                    'temperature_c': 'Temperature (°C)', 'humidity_percent': 'Humidity (%)'},
                color_continuous_scale='Blues'
            )
            fig_scatter1.update_layout(height=400)
            st.plotly_chart(fig_scatter1, use_container_width=True)

        with col2:
            fig_scatter2 = px.scatter(
                filtered_df,
                x='sunshine_hours',
                y='temperature_c',
                color='cloud_cover_percent',
                title='Temperature vs Sunshine (colored by Cloud Cover)',
                labels={
                    'temperature_c': 'Temperature (°C)', 'sunshine_hours': 'Sunshine Hours'},
                color_continuous_scale='RdYlBu_r'
            )
            fig_scatter2.update_layout(height=400)
            st.plotly_chart(fig_scatter2, use_container_width=True)

    with tab3:
        st.subheader("Correlation Analysis")

        # Correlation heatmap
        corr_vars = ['temperature_c', 'humidity_percent', 'pressure_hpa',
                     'wind_speed_kmph', 'cloud_cover_percent', 'rainfall_mm', 'sunshine_hours']
        corr_matrix = filtered_df[corr_vars].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Correlation Matrix of Weather Variables',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlations with temperature
        st.markdown("#### Correlations with Temperature")
        temp_corr = corr_matrix['temperature_c'].sort_values(ascending=False)
        temp_corr = temp_corr.drop('temperature_c')

        fig_bar = px.bar(
            x=temp_corr.values,
            y=temp_corr.index,
            orientation='h',
            title='Feature Correlations with Temperature',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
            color=temp_corr.values,
            color_continuous_scale='RdYlGn'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Seasonal Analysis")

        # Monthly averages
        monthly_avg = filtered_df.groupby('month_name').agg({
            'temperature_c': 'mean',
            'humidity_percent': 'mean',
            'rainfall_mm': 'mean',
            'sunshine_hours': 'mean'
        }).reindex(months)

        col1, col2 = st.columns(2)

        with col1:
            fig_monthly = px.bar(
                monthly_avg,
                y='temperature_c',
                title='Average Temperature by Month',
                labels={'value': 'Temperature (°C)', 'month_name': 'Month'}
            )
            fig_monthly.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            fig_rainfall = px.bar(
                monthly_avg,
                y='rainfall_mm',
                title='Average Rainfall by Month',
                labels={'value': 'Rainfall (mm)', 'month_name': 'Month'},
                color='rainfall_mm',
                color_continuous_scale='Blues'
            )
            fig_rainfall.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_rainfall, use_container_width=True)

        # Yearly comparison
        yearly_avg = filtered_df.groupby('year').agg({
            'temperature_c': 'mean',
            'rainfall_mm': 'sum',
            'sunshine_hours': 'sum'
        }).reset_index()

        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Bar(
            x=yearly_avg['year'],
            y=yearly_avg['temperature_c'],
            name='Avg Temperature (°C)',
            yaxis='y',
            marker_color='indianred'
        ))
        fig_yearly.add_trace(go.Scatter(
            x=yearly_avg['year'],
            y=yearly_avg['rainfall_mm'],
            name='Total Rainfall (mm)',
            yaxis='y2',
            marker_color='blue'
        ))

        fig_yearly.update_layout(
            title='Yearly Temperature and Rainfall Comparison',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Average Temperature (°C)', side='left'),
            yaxis2=dict(title='Total Rainfall (mm)',
                        overlaying='y', side='right'),
            height=450,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    # Data table
    st.markdown("---")
    st.subheader("📋 Filtered Data")

    show_data = st.checkbox("Show raw data")
    if show_data:
        st.dataframe(
            filtered_df.sort_values('date', ascending=False),
            use_container_width=True,
            height=400
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"temperature_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        f"**Data Range:** {filtered_df['date'].min().strftime('%Y-%m-%d')} to "
        f"{filtered_df['date'].max().strftime('%Y-%m-%d')} | "
        f"**Total Days:** {len(filtered_df)} days"
    )

except FileNotFoundError:
    st.error("❌ Error: Could not find 'Temprature_data.csv'. Please ensure the file is in the correct directory.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
