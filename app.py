import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
import os
import model

# Streamlit config must be the first command
st.set_page_config(layout="wide", page_title="ClearCoast AI", page_icon="🌊")

# --- Custom Styling (Modern Coastal Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f8ff; /* Alice Blue - light background */
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #003366 !important; /* Deep Blue */
        font-weight: 800 !important;
    }
    .subtitle {
        color: #008080; /* Teal */
        font-size: 1.25rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #008080 !important; /* Teal */
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton>button:hover {
        background-color: #005959 !important; /* Darker Teal */
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 3rem;
        background: #ffffff;
        color: #003366;
        border-top: 2px solid #008080;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .alert-box {
        background-color: #e0f2f1; /* Light Teal */
        border-left: 4px solid #008080; /* Teal */
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #004d40;
    }
    /* Style headers for columns */
    h3 {
        color: #003366 !important; /* Deep Blue */
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌊 ClearCoast AI")
st.markdown("<div class='subtitle'>AI-Powered Cloud Removal & Hallucination</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Controls")
use_test_patch = st.sidebar.button("🚀 Use Sample Cloudy Image", use_container_width=True)

st.sidebar.markdown("---")
# The upload limit is controlled via config.toml, but we indicate it here
uploaded_file = st.sidebar.file_uploader("Upload Image (Max 500 MB)", type=["png", "jpg", "jpeg", "tif", "tiff"])

@st.cache_resource
def load_app_model():
    return model.get_model()

def load_local_sample():
    target_file = "chennai_s2_cloudy_2026-02-13.tif"
    if os.path.exists(target_file):
        img_array = np.array(Image.open(target_file).convert("RGB"))
        return cv2.resize(img_array, (1024, 1024), interpolation=cv2.INTER_AREA)
    
    # Error out if the file isn't present
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
            
            # Make processing faster by resizing input to 1024x1024
            img_array = cv2.resize(img_array, (1024, 1024), interpolation=cv2.INTER_AREA)

        # Baseline optical stretching
        img_flat = img_array.reshape(-1, 3)
        p2, p98 = np.percentile(img_flat, (2, 98), axis=0)
        p2[p2 == p98] = 0 
        img_display_float = np.clip((img_array - p2) / (p98 - p2 + 1e-8), 0, 1)
        img_display_uint8 = (img_display_float * 255).astype(np.uint8)

        # Process via model
        net = load_app_model()
        out_img_np, conf_map, dynamic_alerts = net.process(img_display_uint8)
    
    # --- Image Display ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original Image (with Clouds)")
        st.image(img_display_uint8, use_container_width=True)
        
    with col2:
        st.subheader("AI Hallucinated Clear View")
        st.image(out_img_np, use_container_width=True)
        
    # --- Analytics Display ---
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

else:
    st.info("👈 Please upload an image or click '🚀 Use Sample Cloudy Image' to begin.")

# --- Footer ---
st.markdown("<div class='footer'>Developed for Academic Project Review | ClearCoast AI</div>", unsafe_allow_html=True)
