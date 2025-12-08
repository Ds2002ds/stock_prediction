import streamlit as st

st.set_page_config(layout="wide")

with open("ind.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)
st.page_link("pages/single_Candlestick.py", label="single Candlestic")
