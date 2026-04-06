import numpy as np
import cv2
import math

class GeneralCloudRemover:
    """
    Advanced Multi-Scale Simulation incorporating intelligent HSV+Variance cloud isolation,
    adaptive localized Dehazing, highly targeted Telea inpainting avoiding land smudging,
    and terrestrial color corrections to perfectly hallucinate clean geographies smoothly.
    """
    def process(self, img_array):
        # Base Conversions
        img_float = img_array.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        
        # 1. Intelligent Cloud Detection (Brightness + Saturation + Local Variance)
        # Calculate local variance of brightness to distinguish clouds from bright sand/urban areas
        v_blur = cv2.GaussianBlur(v, (11, 11), 0)
        v_std = np.sqrt(cv2.GaussianBlur((v - v_blur)**2, (11, 11), 0) + 1e-6)
        
        # Clouds are intensely bright, have low color saturation, and have smoother internal variance than cities
        heavy_cloud = (v > 0.85) & (s < 0.25) & (v_std < 0.15)
        moderate_cloud = (v > 0.65) & (s < 0.40)
        
        # 2. Adaptive Dehazing (DCP)
        dark_channel = cv2.erode(np.min(img_float, axis=2), np.ones((7, 7))) 
        # Smooth the dark channel broadly to prevent blocky artifacts
        dark_channel = cv2.GaussianBlur(dark_channel, (31, 31), 0) 
        
        flat_dark = dark_channel.reshape(-1)
        flat_img = img_float.reshape(-1, 3)
        num_pixels = max(int(len(flat_dark) * 0.001), 1)
        
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        A = np.mean(flat_img[indices], axis=0)
        A = np.clip(A, 0.4, 1.0)
        
        # Transmission map (Backed off omega to 0.85 to prevent over-darkening artifacts)
        t = 1.0 - 0.85 * (dark_channel / np.max(A))
        t = np.clip(t, 0.1, 1.0)
        
        dehazed = (img_float - A) / np.expand_dims(t, 2) + A
        dehazed = np.clip(dehazed, 0, 1)
        dehazed_uint8 = (dehazed * 255).astype(np.uint8)
        
        # 3. Targeted TELEA Inpainting (Hallucination)
        heavy_cloud_uint8 = (heavy_cloud * 255).astype(np.uint8)
        # Only slight dilation (iterations=1) to prevent the inpaint from eating into valid surrounding geography
        heavy_cloud_mask = cv2.dilate(heavy_cloud_uint8, np.ones((5, 5), np.uint8), iterations=1)
        
        if np.sum(heavy_cloud_mask) > 0:
            # Radius 5 provides natural filling without excessive smudging/blurring of landmasses
            inpainted = cv2.inpaint(dehazed_uint8, heavy_cloud_mask, 5, cv2.INPAINT_TELEA)
        else:
            inpainted = dehazed_uint8
            
        # 4. Multi-Scale Detail Enhancement (Fixing inherent inpainting and dehazing softness)
        # Scale 1: Fine micro-textures (buildings, ripples)
        blur1 = cv2.GaussianBlur(inpainted, (3, 3), 1.0)
        sharp1 = cv2.addWeighted(inpainted, 1.5, blur1, -0.5, 0) # Gentle, non-aggressive sharpening
        
        # Scale 2: Medium biological structures / contrast
        blur2 = cv2.GaussianBlur(sharp1, (9, 9), 2.0)
        sharp2 = cv2.addWeighted(sharp1, 1.2, blur2, -0.2, 0) 
        
        # 5. Gentle Coastal Color Correction
        final_hsv = cv2.cvtColor(sharp2, cv2.COLOR_RGB2HSV).astype(np.float32)
        final_h, final_s, final_v = cv2.split(final_hsv)
        
        # Natural broad saturation lift
        final_s = final_s * 1.15 
        
        # Deep Water (Gentle blue restoration)
        water_mask = (final_h > 90) & (final_h < 135)
        final_v[water_mask] = final_v[water_mask] * 0.90 
        final_s[water_mask] = final_s[water_mask] * 1.20
        
        # Landmass (Gentle green/brown warmth)
        land_mask = (final_h > 20) & (final_h < 85)
        final_s[land_mask] = final_s[land_mask] * 1.10
        
        final_hsv[:,:,1] = np.clip(final_s, 0, 255)
        final_hsv[:,:,2] = np.clip(final_v, 0, 255)
        
        final_img = cv2.cvtColor(final_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 6. Improved organic Confidence Map classification
        conf_map = np.ones(v.shape, dtype=np.float32)
        conf_map[moderate_cloud] = 0.5 # Orange
        conf_map[heavy_cloud_mask > 0] = 0.0 # Red
        # Smooth organic gradient
        conf_map = cv2.GaussianBlur(conf_map, (31, 31), 0)
        
        # 7. Alerts
        alerts = []
        if np.mean(land_mask) > 0.05:
            alerts.append("High Vegetation Detected 🌿")
        if np.mean(water_mask) > 0.10:
            alerts.append("Coastal/Water Body 🌊")
        if len(alerts) == 0:
            alerts.append("Urban/Barren Area Clear 🏙️")
            
        return final_img, conf_map, alerts

def get_model():
    return GeneralCloudRemover()
