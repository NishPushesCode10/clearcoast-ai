import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
import os
import google.generativeai as genai

# Streamlit config must be the first command
st.set_page_config(layout="wide", page_title="ClearCoast AI", page_icon="🌊")

# === GOOGLE GEMINI INTEGRATION ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- Custom Styling (Modern Coastal Theme) ---
st.markdown("""
<style>
    .stApp { background-color: #f0f8ff; font-family: 'Inter', sans-serif; }
    h1 { color: #003366 !important; font-weight: 800 !important; }
    .subtitle { color: #008080; font-size: 1.25rem; font-weight: 500; margin-bottom: 2rem; }
    .stButton>button { background-color: #008080 !important; color: white !important; border-radius: 8px !important; }
    .footer { text-align: center; padding: 1rem; margin-top: 3rem; background: #ffffff; color: #003366; border-top: 2px solid #008080; border-radius: 4px; font-size: 0.9rem; }
    .alert-box { background-color: #e0f2f1; border-left: 4px solid #008080; padding: 10px; border-radius: 4px; margin-bottom: 10px; color: #004d40; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌊 ClearCoast AI")
st.markdown("<div class='subtitle'>AI-Powered Cloud Removal & Hallucination</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Controls")
use_test_patch = st.sidebar.button("🚀 Use Sample Cloudy Image", use_container_width=True)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Image (Max 500 MB)", type=["png", "jpg", "jpeg", "tif", "tiff"])

@st.cache_resource
def load_app_model():
    return model.get_model()

def load_local_sample():
    target_file = "chennai_s2_cloudy_2026-02-13.tif"
    if os.path.exists(target_file):
        img_array = np.array(Image.open(target_file).convert("RGB"))
        return cv2.resize(img_array, (1024, 1024), interpolation=cv2.INTER_AREA)
    st.error(f"Sample image '{target_file}' not found. Please upload an image.")
    st.stop()

# --- Main Logic ---
if use_test_patch:
    st.session_state.test_mode = True
if uploaded_file is not None:
    st.session_state.test_mode = False

if st.session_state.get("test_mode", False) or uploaded_file is not None:
    with st.spinner("Processing image..."):
        if st.session_state.get("test_mode", False):
            img_array = load_local_sample()
        else:
            bytes_data = uploaded_file.getvalue()
            img_array = np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))
            img_array = cv2.resize(img_array, (1024, 1024), interpolation=cv2.INTER_AREA)

        net = load_app_model()
        out_img_np, conf_map, dynamic_alerts = net.process(img_array)

    # --- Image Display ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original Image (with Clouds)")
        st.image(img_array, use_container_width=True)
    with col2:
        st.subheader("AI Hallucinated Clear View")
        st.image(out_img_np, use_container_width=True)

    # --- Analytics ---
    st.markdown("---")
    col3, col4 = st.columns([1.5, 1], gap="large")
    with col3:
        st.subheader("Confidence Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('Confidence', rotation=-90, va="bottom")
        st.pyplot(fig)
        st.caption("🟢 Green: High Confidence  |  🔴 Red: AI Hallucinated Regions")

    with col4:
        st.subheader("Simple Alerts")
        if dynamic_alerts:
            for alert in dynamic_alerts:
                st.markdown(f"<div class='alert-box'>{alert}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-box'>No active alerts.</div>", unsafe_allow_html=True)

    # === GEMINI GENAI SECTION ===
    st.markdown("---")
    st.subheader("🤖 GenAI Coastal Insights (Gemini)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if st.button("📊 Generate AI Report"):
            with st.spinner("Analyzing with Gemini..."):
                prompt = f"""
                You are a coastal monitoring expert for Tamil Nadu.
                Analyze the following processed satellite image:
                - NDVI-based alerts: {dynamic_alerts if 'dynamic_alerts' in locals() else 'No alerts available'}
                - Average confidence score: {conf_map.mean() if 'conf_map' in locals() else 0.75:.2f}
                Provide a short, professional report suitable for coastal authorities.
                """
                response = gemini_model.generate_content(prompt)
                st.success("**AI Generated Report:**")
                st.write(response.text)

    with col_g2:
        st.subheader("💬 Ask About the Coast")
        user_question = st.text_input("Ask any question about the current image...")
        if user_question:
            with st.spinner("Thinking with Gemini..."):
                response = gemini_model.generate_content(user_question)
                st.write(response.text)

else:
    st.info("👈 Please upload an image or click '🚀 Use Sample Cloudy Image' to begin.")

# --- Footer ---
st.markdown("<div class='footer'>Developed for Academic Project Review | ClearCoast AI</div>", unsafe_allow_html=True)
