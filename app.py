# -------------------------------
# Background Styling - Dark Sleek
# -------------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.9);
    backdrop-filter: blur(6px);
    border-right: 2px solid #0f2027;
}

/* Headings */
h1, h2, h3, h4 {
    color: #66ccff;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #1e90ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 0 10px #1e90ff;
}
.stButton > button:hover {
    background-color: #005f99;
    box-shadow: 0 0 15px #00bfff;
}
</style>
""", unsafe_allow_html=True)
