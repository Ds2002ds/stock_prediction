import streamlit as st

st.set_page_config(layout="wide")

query = st.query_params
page = query.get("page", ["home"])[0]

if page == "single":
    st.switch_page("pages/Single Candlestick.py")

elif page == "double":
    st.switch_page("pages/Double_Candlestick.py")

else:
    st.title("🏠 Home")
    with open("ind.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=800, scrolling=True)
