import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Interactive EDA Dashboard", layout="wide")

# Custom Styling
st.markdown("<h1 style='text-align: center; color: blue;'>📊 Interactive EDA Dashboard</h1>", unsafe_allow_html=True)

# Dataset Selection
dataset_option = st.sidebar.selectbox("Choose a dataset:", ["Titanic", "Iris", "Real-Time Stocks"])

# Load dataset
if dataset_option == "Titanic":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
elif dataset_option == "Iris":
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
else:
    url = None

# Load Data
if dataset_option in ["Titanic", "Iris"]:
    data = pd.read_csv(url)
    data.dropna(inplace=True)  # Drop missing values

    # Display Data
    st.write("### 🔍 Dataset Preview")
    st.write(data.head())

    # Summary Statistics
    st.write("### 📊 Summary Statistics")
    st.write(data.describe())

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(data.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatter Plot
    st.write("### ✨ Scatter Plot")
    fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color=data.columns[-1])
    st.plotly_chart(fig)

    # Box Plot
    st.write("### 🎭 Box Plot")
    fig = px.box(data, x=data.columns[-1], y=data.columns[1])
    st.plotly_chart(fig)

    # K-Means Clustering
    st.write("### 🤖 K-Means Clustering")
    num_clusters = st.slider("Select number of clusters:", 2, 10, 3)

    scaler = StandardScaler()
    X = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)

    fig = px.scatter(data, x=data.columns[1], y=data.columns[2], color=data['Cluster'].astype(str),
                     title="K-Means Clustering")
    st.plotly_chart(fig)

# Real-Time Stock Data
elif dataset_option == "Real-Time Stocks":
    st.write("### 📈 Real-Time Stock Market Analysis")
    
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL):", "AAPL")
    stock_data = yf.Ticker(stock_symbol).history(period="1mo")

    # Display Data
    st.write(f"### 🏦 Stock Data for {stock_symbol}")
    st.write(stock_data.tail())

    # Line Chart for Closing Price
    st.write("### 📉 Closing Price Trend")
    fig = px.line(stock_data, x=stock_data.index, y="Close", title=f"{stock_symbol} Closing Price")
    st.plotly_chart(fig)

    # Candlestick Chart
    st.write("### 📊 Candlestick Chart")
    fig = px.scatter(stock_data, x=stock_data.index, y="Close", color=stock_data["Close"], title="Stock Price Movement")
    st.plotly_chart(fig)

    # Moving Average
    stock_data["MA10"] = stock_data["Close"].rolling(window=10).mean()
    stock_data["MA20"] = stock_data["Close"].rolling(window=20).mean()

    st.write("### 🔄 Moving Averages")
    fig = px.line(stock_data, x=stock_data.index, y=["Close", "MA10", "MA20"],
                  title=f"{stock_symbol} with Moving Averages")
    st.plotly_chart(fig)

else:
    st.write("Please select a dataset to explore.")
