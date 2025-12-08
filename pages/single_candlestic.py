import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Double Candlestick", layout="wide")
st.title("📊 single Candlestick Pattern")

html_file = Path("single_candlestick.html")

if html_file.exists():
    html_content = html_file.read_text(encoding="utf-8")
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.error("HTML file not found")
