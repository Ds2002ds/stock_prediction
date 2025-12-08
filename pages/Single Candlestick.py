import streamlit as st

st.set_page_config(layout="wide")

st.title("🏠 Home")

with open("ind.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)
