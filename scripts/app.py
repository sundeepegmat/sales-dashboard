import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="My Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 My First Streamlit Dashboard")
st.markdown("Welcome to your basic web dashboard!")

# Sidebar
st.sidebar.header("Dashboard Controls")
chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ["Line Chart", "Bar Chart", "Scatter Plot"]
)

# Create sample data
@st.cache_data
def load_data():
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 1000, 100),
        'Customers': np.random.randint(50, 200, 100),
        'Revenue': np.random.randint(1000, 5000, 100)
    })
    return data

data = load_data()

# Main dashboard layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Sales",
        value=f"{data['Sales'].sum():,}",
        delta=f"{data['Sales'].tail(7).sum() - data['Sales'].head(7).sum()}"
    )

with col2:
    st.metric(
        label="Average Customers",
        value=f"{data['Customers'].mean():.0f}",
        delta=f"{data['Customers'].tail(7).mean() - data['Customers'].head(7).mean():.0f}"
    )

with col3:
    st.metric(
        label="Total Revenue",
        value=f"${data['Revenue'].sum():,}",
        delta=f"{data['Revenue'].tail(7).sum() - data['Revenue'].head(7).sum()}"
    )

# Charts section
st.subheader("📈 Data Visualization")

if chart_type == "Line Chart":
    fig = px.line(data, x='Date', y='Sales', title='Sales Over Time')
    st.plotly_chart(fig, use_container_width=True)
elif chart_type == "Bar Chart":
    # Group by month for bar chart
    monthly_data = data.groupby(data['Date'].dt.to_period('M'))['Sales'].sum()
    fig = px.bar(x=monthly_data.index.astype(str), y=monthly_data.values, 
                 title='Monthly Sales', labels={'x': 'Month', 'y': 'Sales'})
    st.plotly_chart(fig, use_container_width=True)
else:  # Scatter Plot
    fig = px.scatter(data, x='Customers', y='Revenue', 
                     title='Customers vs Revenue', 
                     hover_data=['Date'])
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader("📋 Raw Data")
if st.checkbox("Show raw data"):
    st.dataframe(data, use_container_width=True)

# Interactive filters
st.subheader("🔍 Data Filters")
col1, col2 = st.columns(2)

with col1:
    date_range = st.date_input(
        "Select Date Range",
        value=(data['Date'].min(), data['Date'].max()),
        min_value=data['Date'].min(),
        max_value=data['Date'].max()
    )

with col2:
    sales_range = st.slider(
        "Sales Range",
        min_value=int(data['Sales'].min()),
        max_value=int(data['Sales'].max()),
        value=(int(data['Sales'].min()), int(data['Sales'].max()))
    )

# Filter data based on selections
filtered_data = data[
    (data['Date'] >= pd.to_datetime(date_range[0])) & 
    (data['Date'] <= pd.to_datetime(date_range[1])) &
    (data['Sales'] >= sales_range[0]) & 
    (data['Sales'] <= sales_range[1])
]

st.write(f"Filtered data: {len(filtered_data)} rows")
st.dataframe(filtered_data.head(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🚀")