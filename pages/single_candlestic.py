import streamlit as st

st.set_page_config(
    page_title="Single Candlestick Pattern",
    layout="wide"
)

st.title("📊 Single Candlestick Patterns")

# Optional Home navigation
st.page_link("app.py", label="🏠 Home")

st.markdown("---")

# ---------- Helper function ----------
def pattern(title, description, points, strategy, image=None):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(title)
        st.write(description)

        st.markdown("**Key Points:**")
        for p in points:
            st.write(f"• {p}")

        st.markdown("**Trading Strategy:**")
        for s in strategy:
            st.write(f"• {s}")

    with col2:
        if image:
            st.image(image, use_container_width=True)

    st.markdown("---")


# ✅ HAMMER
pattern(
    "Hammer Pattern",
    "Hammer is a **bullish reversal** pattern appearing after a downtrend.",
    [
        "Small body near top",
        "Long lower shadow",
        "Little or no upper shadow",
    ],
    [
        "Entry above hammer high",
        "Stop-loss below low",
        "Target near resistance",
    ],
    "images/hammer.png"
)

# ✅ DOJI
pattern(
    "Doji Pattern",
    "Doji shows **market indecision** and possible reversal.",
    [
        "Open and close almost equal",
        "Can appear at tops or bottoms",
    ],
    [
        "Trade after confirmation candle",
        "Use support & resistance",
    ],
    "images/doji.png"
)

# ✅ INVERTED HAMMER
pattern(
    "Inverted Hammer",
    "Bullish reversal pattern after downtrend.",
    [
        "Small body at bottom",
        "Long upper shadow",
    ],
    [
        "Buy above high",
        "Stop-loss below low",
    ],
    "images/inverted_hammer.png"
)

# ✅ FOOTER
st.caption("© 2025 Trade Winds. All rights reserved.")
