# 🎯 Customer Segmentation Streamlit Dashboard

An interactive web application for customer segmentation analysis and prediction using Streamlit.

## 🚀 Features

### 🏠 Home Page
- Quick overview of customer segments
- Key business metrics at a glance
- Segment distribution visualization

### 🔮 Predict Segment
- **Interactive Form**: Input customer details through an intuitive interface
- **Real-time Predictions**: Get instant segment predictions with confidence scores
- **Personalized Recommendations**: Receive tailored marketing strategies for each customer
- **Customer Profiling**: View income level, spending patterns, and engagement metrics

### 📊 Analytics Dashboard
- **Distribution Analysis**: Pie charts and bar graphs showing segment distribution
- **Revenue Analysis**: Total and average revenue by segment
- **Demographics**: Age distribution and engagement metrics across segments
- Interactive Plotly charts for better data exploration

### 👥 Customer Analysis
- **Segment Deep Dive**: Detailed analysis of each customer segment
- **Demographics Breakdown**: Age, gender, and behavioral patterns
- **Spending Behavior**: Order values, purchase frequency, and app engagement
- **Sample Customers**: View actual customer data for each segment

### 📈 Business Insights
- **Strategic Recommendations**: Actionable marketing strategies per segment
- **Growth Opportunities**: Identify areas for revenue and customer base expansion
- **ROI Analysis**: Understand value and investment priorities
- **Segment Comparison**: Side-by-side metrics across all segments

## 📦 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Verify model files exist:**
   ```
   models/
   ├── kmeans_customer_segmentation.pkl
   ├── scaler.pkl
   └── label_encodings.pkl
   ```

3. **Ensure data file exists:**
   ```
   CustomerData_clean.csv
   ```

## 🎮 Usage

### Start the Dashboard

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

### Alternative: Specify Port

```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🎯 Customer Segments

### 💰 Price-Sensitive Occasional
- **Profile**: Low-income, minimal engagement
- **Size**: ~21% of customer base
- **CLV**: ₹54K average
- **Strategy**: Discounts, free shipping, entry-level loyalty

### 👑 High-Value Loyal
- **Profile**: Premium customers, highest engagement
- **Size**: ~17% of customer base
- **CLV**: ₹663K average (highest!)
- **Strategy**: VIP benefits, early access, premium service

### ⭐ Value-Seeking Regular
- **Profile**: Mid-tier, regular engagement
- **Size**: ~62% of customer base (largest!)
- **CLV**: ₹260K average
- **Strategy**: Seasonal sales, bundles, referral programs

## 📊 Dashboard Components

### Sidebar Navigation
- Quick metrics display
- Page navigation
- Model status indicator

### Interactive Elements
- **Forms**: Dynamic input fields with validation
- **Sliders**: Numeric inputs with constraints
- **Dropdowns**: Category selection
- **Buttons**: Trigger predictions and actions

### Visualizations
- **Plotly Charts**: Interactive, zoomable graphs
- **Metrics Cards**: Key performance indicators
- **Data Tables**: Paginated customer data
- **Color-coded Segments**: Visual distinction between groups

## 🎨 Customization

### Modify Segment Information
Edit `SEGMENT_INFO` dictionary in `streamlit_app.py`:
```python
SEGMENT_INFO = {
    0: {
        'name': 'Your Segment Name',
        'color': '#HEX_COLOR',
        'icon': '🎯',
        'description': 'Segment description',
        'characteristics': [...],
        'recommendations': [...]
    }
}
```

### Add New Pages
1. Create page function: `def show_your_page(df):`
2. Add to navigation: Update sidebar radio buttons
3. Add routing: Update main() function if-else logic

### Custom Styling
Modify CSS in `st.markdown()` sections:
```python
st.markdown("""
    <style>
    .your-class {
        /* Your custom CSS */
    }
    </style>
""", unsafe_allow_html=True)
```

## 🔧 Configuration

### Streamlit Settings
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = false
```

### Model Caching
Models are cached using `@st.cache_resource` for performance. Clear cache:
```python
st.cache_resource.clear()
```

## 📱 Features Breakdown

### Prediction Flow
1. User inputs customer details
2. Data is encoded using saved label encodings
3. Features are scaled using saved StandardScaler
4. KMeans model predicts cluster
5. Confidence calculated from cluster distances
6. Results displayed with recommendations

### Analytics Flow
1. Load customer data from CSV
2. Apply clustering if not present
3. Aggregate metrics by segment
4. Generate interactive visualizations
5. Display insights and comparisons

## 🚀 Deployment Options

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements_streamlit.txt
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🐛 Troubleshooting

### Models Not Loading
- Check if `models/` directory exists
- Verify pickle files are not corrupted
- Ensure scikit-learn version matches

### Data Not Displaying
- Confirm `CustomerData_clean.csv` exists
- Check CSV encoding (should be UTF-8)
- Verify column names match expected format

### Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Clear Cache
```bash
streamlit cache clear
```

## 📈 Performance Tips

1. **Use Caching**: Applied to model loading and data loading
2. **Lazy Loading**: Data loaded only when needed
3. **Efficient Filtering**: Use pandas operations for fast filtering
4. **Plotly Over Matplotlib**: Interactive and faster rendering

## 🔐 Security Notes

- No authentication implemented (add for production)
- Models loaded from local files (consider encryption)
- Input validation basic (enhance for production)
- No rate limiting (add for public deployment)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [scikit-learn Guide](https://scikit-learn.org/stable/)

## 🎯 Next Steps

1. Add user authentication
2. Implement prediction history tracking
3. Create downloadable reports (PDF/Excel)
4. Add A/B testing for recommendations
5. Integrate with CRM systems
6. Add real-time data updates

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: February 5, 2026
