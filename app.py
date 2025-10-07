# -------------------------------
# Background Styling - Light Academic
# -------------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #f0f9ff, #e6f7ff, #ffffff);
    color: #000000;
}

/* Sidebar with frosted glass effect */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border-right: 2px solid #e6f2ff;
}

/* Headings */
h1, h2, h3, h4 {
    color: #003366;
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background-color: #0066cc;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #004c99;
}
</style>
""", unsafe_allow_html=True)
