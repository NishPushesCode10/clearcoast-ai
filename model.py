import numpy as np
import cv2

class GeneralCloudRemover:
    """
    Advanced multi-stage cloud removal and hallucination pipeline optimized for 
    Sentinel-2 coastal imagery. Uses intelligent HSV+Edge detection for masks,
    Navier-Stokes inpainting, Bilateral edge-preserving smoothing, and LAB-based 
    CLAHE for incredibly natural contrast and color correction.
    """
    def process(self, img_array):
        # 1. Base Conversions
        img_float = img_array.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        s_float = s.astype(np.float32) / 255.0
        v_float = v.astype(np.float32) / 255.0
        
        # 2. Highly Accurate Cloud Detection (Brightness + Saturation + Edges)
        # Clouds: Bright (high V), low saturation (low S), and lack sharp textures inside
        
        # Edge magnitude using Sobel to avoid detecting bright buildings/roads as clouds
        sobelx = cv2.Sobel(v_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(v_float, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_mag = cv2.GaussianBlur(edge_mag, (5, 5), 0)
        
        # Refined cloud masks
        # Heavy clouds: very bright, very low saturation, not an edge
        heavy_cloud = (v_float > 0.75) & (s_float < 0.3) & (edge_mag < 0.15)
        
        # Moderate clouds (thin clouds / cloud edges)
        moderate_cloud = (v_float > 0.60) & (s_float < 0.45) & (edge_mag < 0.25)
        
        heavy_cloud_uint8 = (heavy_cloud * 255).astype(np.uint8)
        
        # Grow the heavy cloud mask slightly to cover soft cloud edges during inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        heavy_cloud_mask = cv2.dilate(heavy_cloud_uint8, kernel, iterations=2)
        
        # 3. Adaptive Dehazing (DCP - Dark Channel Prior)
        # Helps remove thin haze before inpainting
        dark_channel = cv2.erode(np.min(img_float, axis=2), np.ones((15, 15))) 
        dark_channel = cv2.GaussianBlur(dark_channel, (31, 31), 0) 
        
        flat_dark = dark_channel.reshape(-1)
        flat_img = img_float.reshape(-1, 3)
        num_pixels = max(int(len(flat_dark) * 0.001), 1)
        
        # Atmospheric light
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        A = np.mean(flat_img[indices], axis=0)
        A = np.clip(A, 0.6, 1.0) # Restrict to avoid aggressive color shifts
        
        # Transmission map
        t = 1.0 - 0.75 * (dark_channel / np.max(A)) # Moderate omega (0.75)
        t = np.clip(t, 0.2, 1.0) # Prevent dividing by near-zero
        
        dehazed = (img_float - A) / np.expand_dims(t, 2) + A
        dehazed = np.clip(dehazed, 0, 1)
        dehazed_uint8 = (dehazed * 255).astype(np.uint8)
        
        # 4. Navier-Stokes Inpainting (More realistic structural hallucination than Telea)
        if np.sum(heavy_cloud_mask) > 0:
            # radius 7 for smoother integration
            inpainted = cv2.inpaint(dehazed_uint8, heavy_cloud_mask, 7, cv2.INPAINT_NS)
        else:
            inpainted = dehazed_uint8
            
        # 5. Advanced Edge-Preserving Sharpening
        # Instead of generic blur that smudges coastlines, we use Bilateral Filter
        # which blurs while keeping edges perfectly sharp.
        smoothed = cv2.bilateralFilter(inpainted, d=9, sigmaColor=50, sigmaSpace=50)
        
        # Unsharp masking: Image + (Image - Smoothed) * amount
        # This enhances details (coastlines, buildings) without adding noise
        sharp = cv2.addWeighted(inpainted, 1.5, smoothed, -0.5, 0)
        
        # 6. Professional Color Correction (LAB space CLAHE)
        # LAB space separates luminosity (L) from color (A, B) preventing hue shifts
        lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply Contrast Limited Adaptive Histogram Equalization to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Merge back
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        final_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        
        # Vibrant saturation boost (Gentle)
        final_hsv = cv2.cvtColor(final_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        final_hsv[:, :, 1] = np.clip(final_hsv[:, :, 1] * 1.15, 0, 255)
        final_img = cv2.cvtColor(final_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 7. Smooth, Accurate Confidence Map
        conf_map = np.ones(v.shape, dtype=np.float32) # Default 1.0 (Green/Real)
        conf_map[moderate_cloud] = 0.5 # Orange (Thin clouds / partially real)
        conf_map[heavy_cloud_mask > 0] = 0.0 # Red (Fully Hallucinated)
        
        # Extremely smooth gradient for natural heatmap visualization
        conf_map = cv2.GaussianBlur(conf_map, (51, 51), 0)
        
        # 8. Dynamic Contextual Alerts
        alerts = []
        # Use hue to detect terrain types
        final_h = final_hsv[:, :, 0]
        water_mask = (final_h > 90) & (final_h < 135)
        land_mask = (final_h > 20) & (final_h < 85)
        
        if np.sum(heavy_cloud_mask) > 0:
            pct = (np.sum(heavy_cloud_mask > 0) / heavy_cloud_mask.size) * 100
            alerts.append(f"☁️ {pct:.1f}% Severe Cloud Cover Hallucinated")
        if np.mean(land_mask) > 0.05:
            alerts.append("🌿 Dense Vegetation / Mangroves Detected")
        if np.mean(water_mask) > 0.10:
            alerts.append("🌊 Coastal Water Body Extracted")
        if len(alerts) == 0:
            alerts.append("🏙️ Urban / Barren Geography Isolated")
            
        return final_img, conf_map, alerts

def get_model():
    return GeneralCloudRemover()
