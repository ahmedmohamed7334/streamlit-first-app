import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import time

# Set Streamlit Page Config
st.set_page_config(page_title="Interactive EDA Dashboard", layout="wide")

# Sidebar - Dataset Selection
dataset_option = st.sidebar.selectbox("Choose a dataset", ["Titanic", "Iris", "COVID-19 Trends", "Spotify", "Real-Time Stock Data"])

# Load Data
@st.cache_data
def load_data(dataset):
    if dataset == "Titanic":
        return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    elif dataset == "Iris":
        return pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    elif dataset == "COVID-19 Trends":
        return pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    elif dataset == "Spotify":
        return pd.read_csv("https://raw.githubusercontent.com/sonofesh/spotify-data/master/data.csv")
    elif dataset == "Real-Time Stock Data":
        return None  # Handled separately

data = load_data(dataset_option)

# Real-Time Data Handling
if dataset_option == "Real-Time Stock Data":
    st.sidebar.write("Fetching Real-Time Stock Data...")
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
    api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey=demo"
    response = requests.get(api_url).json()
    if "Time Series (5min)" in response:
        stock_df = pd.DataFrame.from_dict(response["Time Series (5min)"], orient='index').astype(float)
        stock_df = stock_df.reset_index().rename(columns={'index': 'Timestamp'})
        st.write("### Real-Time Stock Data for", stock_symbol)
        st.line_chart(stock_df[['1. open', '4. close']])
    else:
        st.error("Invalid stock symbol or API limit reached.")
    st.stop()

# Data Overview
st.write("### Dataset Preview")
st.dataframe(data.head())

# Summary Statistics
st.write("### Summary Statistics")
st.write(data.describe())

# Interactive Filtering
if dataset_option in ["Titanic", "Iris"]:
    column_filter = st.sidebar.selectbox("Filter by Column", data.columns)
    unique_values = data[column_filter].dropna().unique()
    selected_value = st.sidebar.selectbox("Choose a value", unique_values)
    filtered_data = data[data[column_filter] == selected_value]
    st.write("### Filtered Data")
    st.dataframe(filtered_data)

# Visualizations
st.write("### Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    if dataset_option == "Titanic":
        fig = px.histogram(data, x="Age", color="Survived", title="Age Distribution by Survival Status")
    elif dataset_option == "Iris":
        fig = px.scatter(data, x="sepal_length", y="petal_length", color="species", title="Sepal vs Petal Length")
    else:
        fig = px.histogram(data, x=data.columns[1], title="Histogram of Feature")
st.plotly_chart(fig)

with col2:
    if dataset_option == "COVID-19 Trends":
        country = st.sidebar.selectbox("Select Country", data['location'].dropna().unique())
        country_data = data[data['location'] == country]
        fig = px.line(country_data, x="date", y="total_cases", title=f"COVID-19 Cases in {country}")
    else:
        fig = px.box(data, x=data.columns[-1], title="Box Plot of Feature")
st.plotly_chart(fig)

# Machine Learning - Clustering
st.write("### Clustering with K-Means")
num_clusters = st.slider("Select Number of Clusters", 2, 5, 3)
scaler = StandardScaler()
X = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
fig = px.scatter(data, x=data.columns[1], y=data.columns[2], color=data['Cluster'].astype(str), title="K-Means Clustering")
st.plotly_chart(fig)

# Predictive Analysis
st.write("### Predictive Trend Analysis")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[data.columns[1]], mode='lines', name='Feature 1'))
fig.add_trace(go.Scatter(x=data.index, y=data[data.columns[2]], mode='lines', name='Feature 2'))
st.plotly_chart(fig)

st.success("Interactive EDA Dashboard Loaded Successfully!")