

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2
import glob
import model

# Guarantee zero decompression crashes for large generalized media loads on App Service RAM
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(layout="wide", page_title="ClearCoast AI", page_icon="🌊")
st._config.set_option("server.maxUploadSize", 500)

# --- Custom Styling for Premium Lightweight Dashboard ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    h1 { color: #0284c7 !important; font-weight: 800 !important; }
    .subtitle { color: #475569; font-size: 1.25rem; font-weight: 500; margin-bottom: 2rem; }
    .footer { position: relative; text-align: center; padding: 1rem; margin-top: 3rem; background: #0f172a; color: white; border-radius: 6px; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("🌊 ClearCoast AI")
st.markdown("<div class='subtitle'>AI-Powered Cloud Removal & Hallucination</div>", unsafe_allow_html=True)

# --- Sidebar Portal ---
st.sidebar.markdown("## Control Panel")
use_test_patch = st.sidebar.button("🚀 Use Sample Cloudy Image", type="primary", use_container_width=True)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Image (.jpg, .png, .tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])

@st.cache_resource
def load_app_model():
    return model.get_model()

def create_synthetic_landscape():
    """Fallback generator ensuring instant load without heavy disk assets."""
    np.random.seed(42)
    patch = np.zeros((512, 512, 3), dtype=np.uint8)
    patch[:, :] = [100, 150, 200] 
    cv2.fillPoly(patch, [np.array([[0,512], [0,250], [150,150], [300,280], [512, 200], [512,512]])], (34, 100, 34)) 
    cv2.fillPoly(patch, [np.array([[0,512], [0,400], [512, 300], [512,512]])], (30, 100, 170)) 
    
    noise = np.random.randint(-15, 15, (512, 512, 3), dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.circle(patch, (150, 200), 80, (230, 230, 230), -1)
    cv2.circle(patch, (350, 150), 100, (240, 240, 240), -1)
    cv2.circle(patch, (250, 300), 70, (220, 220, 220), -1)
    
    patch = cv2.GaussianBlur(patch, (35, 35), 0) 
    return Image.fromarray(patch)

def load_local_sample():
    """Load the real cloudy Sentinel-2 image for the sample button."""
    try:
        # Load your real cloudy image from the project folder
        img = Image.open("chennai_s2_cloudy_2026-02-13.tif")
        # Resize to 1024x1024 for fast processing on Azure
        img = img.resize((1024, 1024))
        return np.array(img.convert("RGB"))
    except Exception as e:
        st.warning(f"Could not load real sample image: {e}")
        # Fallback synthetic image
        return np.array(create_synthetic_landscape().convert("RGB"))

# --- Session Router ---
if "test_mode" not in st.session_state:
    st.session_state.test_mode = False

if use_test_patch:
    st.session_state.test_mode = True

if uploaded_file is not None:
    st.session_state.test_mode = False

if st.session_state.test_mode or uploaded_file is not None:
    with st.spinner("Executing Lightweight Telea & Pillow AI Pass..."):
        if st.session_state.test_mode:
            img_array = load_local_sample()
        else:
            bytes_data = uploaded_file.getvalue()
            img_array = np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))
            
        # Rigid sizing constraints blocking memory limits on Azure instances safely
        orig_h, orig_w = img_array.shape[:2]
        max_dim = 1024
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Baseline optical stretching
        img_flat = img_array.reshape(-1, 3)
        p2, p98 = np.percentile(img_flat, (2, 98), axis=0)
        p2[p2 == p98] = 0 
        img_display_float = np.clip((img_array - p2) / (p98 - p2 + 1e-8), 0, 1)
        img_display_uint8 = (img_display_float * 255).astype(np.uint8)

        # Stream physical bytes into cv2 process architecture
        net = load_app_model()
        out_img_np, conf_map, dynamic_alerts = net.process(img_display_uint8)
    
    # --- Imagery Board ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write("### Original Image (with Clouds)")
        st.image(img_display_uint8, use_container_width=True)
        
    with col2:
        st.write("### AI Hallucinated Clear View")
        st.image(out_img_np, use_container_width=True)
        
    # --- Analytical Dashboards ---
    st.markdown("---")
    col3, col4 = st.columns([1.5, 1], gap="large")
    
    with col3:
        st.write("### Hallucination Confidence Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('Inferred Accuracy', rotation=-90, va="bottom")
        st.pyplot(fig)
        st.caption("🟢 Green: High Confidence (Real Structures)  |  🔴 Red: AI Hallucinated Boundaries")
        
    with col4:
        st.write("### Extracted Generative Alerts")
        st.caption("Region-level ecological tracking:")
        for alert in dynamic_alerts:
            st.info(f"**{alert}**")

else:
    st.info("👈 **Awaiting Data:** Please upload a generalized media format or click the blue sample button.")

# --- Fixed Footer ---
st.markdown("<div class='footer'>Extremely Fast, Lightweight, Azure-Ready Streamlit Deployment Model</div>", unsafe_allow_html=True)


