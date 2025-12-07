import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Indian Stock Dashboard", layout="wide")

st.title("📊 Indian Stock Market Dashboard")

# Stock selector
stock_symbol = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

# Fetch data
data = yf.download(stock_symbol, period="1d", interval="1m")

if not data.empty:
    last_price = data["Close"].iloc[-1]
    change = last_price - data["Close"].iloc[-2]

    col1, col2, col3 = st.columns(3)
    col1.metric("Stock", stock_symbol)
    col2.metric("Price (₹)", round(last_price, 2), round(change, 2))
    col3.metric("Volume", int(data["Volume"].iloc[-1]))

    st.line_chart(data["Close"])
else:
    st.error("Failed to load data")

st.caption("Data source: Yahoo Finance")
