import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2

# Force higher upload limit
st._config.set_option("server.maxUploadSize", 500)

st.set_page_config(layout="wide", page_title="ClearCoast AI", page_icon="🌊")

# Modern styling
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    h1 { color: #0284c7 !important; font-weight: 800 !important; }
    .subtitle { color: #475569; font-size: 1.25rem; font-weight: 500; margin-bottom: 2rem; }
    .footer { text-align: center; padding: 1rem; margin-top: 3rem; background: #0f172a; color: white; border-radius: 6px; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🌊 ClearCoast AI")
st.markdown("<div class='subtitle'>AI-Powered Cloud Removal & Hallucination</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Control Panel")
use_sample = st.sidebar.button("🚀 Use Sample Cloudy Image", type="primary", use_container_width=True)
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Image (.jpg, .png, .tif)", type=["png", "jpg", "jpeg", "tif", "tiff"])

@st.cache_resource
def load_app_model():
    return model.get_model()

def load_real_sample():
    try:
        img = Image.open("chennai_s2_cloudy_2026-02-13.tif")
        img = img.resize((1024, 1024))  # Faster processing
        return np.array(img.convert("RGB"))
    except:
        st.error("Sample image not found in project folder.")
        return None

if use_sample or uploaded_file is not None:
    with st.spinner("Processing with AI hallucination..."):
        if use_sample:
            img_array = load_real_sample()
        else:
            bytes_data = uploaded_file.getvalue()
            img_array = np.array(Image.open(io.BytesIO(bytes_data)).convert("RGB"))

        # Early resize for speed
        if img_array is not None:
            orig_h, orig_w = img_array.shape[:2]
            max_dim = 1024
            if max(orig_h, orig_w) > max_dim:
                scale = max_dim / max(orig_h, orig_w)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Process with model
            net = load_app_model()
            out_img_np, conf_map, dynamic_alerts = net.process(img_array)

    # Display
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.write("### Original Image (with Clouds)")
        st.image(img_array, use_container_width=True)

    with col2:
        st.write("### AI Hallucinated Clear View")
        st.image(out_img_np, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns([1.5, 1], gap="large")
    with col3:
        st.write("### Hallucination Confidence Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel('Inferred Accuracy', rotation=-90, va="bottom")
        st.pyplot(fig)

    with col4:
        st.write("### Alerts")
        for alert in dynamic_alerts:
            st.info(f"**{alert}**")

else:
    st.info("👈 Please upload an image or click the sample button to begin.")

st.markdown("<div class='footer'>Lightweight • Fast • Azure-Ready</div>", unsafe_allow_html=True)
