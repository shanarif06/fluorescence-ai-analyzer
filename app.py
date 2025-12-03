import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smartphone Sensing Analyzer", layout="wide")
st.title("📱 Smartphone Sensing Analyzer")
st.caption("Extract RGB, analyze with AI, and track your experiments.")

# --- LAYOUT ---
col1, col2 = st.columns([1.2, 1])

# --- API Setup ---
api_key = st.secrets["GEMINI_API_KEY"]
gemini_api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

# --- LEFT COLUMN: Image Upload + ROI ---
with col1:
    uploaded = st.file_uploader("Click to Upload Fluorescence Image", type=["png", "jpg", "jpeg"])
    roi_size = st.slider("Region of Interest (ROI) Size", 5, 50, 15, step=5)
    
    if uploaded:
        image = Image.open(uploaded)
        img = np.array(image)
        h, w, _ = img.shape
        
        st.image(img, caption="Uploaded Image", use_container_width=True)
        x = st.slider("ROI X Position", 0, w-1, w//2)
        y = st.slider("ROI Y Position", 0, h-1, h//2)
        
        # Extract ROI and RGB
        roi = img[max(0, y-roi_size):y+roi_size, max(0, x-roi_size):x+roi_size]
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        r, g, b = [int(c) for c in avg_color]
        
        st.subheader("🎨 RGB Values")
        c1, c2, c3 = st.columns(3)
        c1.metric("R", r)
        c2.metric("G", g)
        c3.metric("B", b)
    else:
        st.info("Upload an image to start analysis.")

# --- RIGHT COLUMN: AI Analysis ---
with col2:
    st.header("⚡ AI-Enhanced Prediction")

    if uploaded:
        if st.button("Estimate Intensity"):
            st.write("Analyzing with AI model... ⏳")
            prompt = f"""
            You are a fluorescence intensity analyzer. 
            Estimate relative fluorescence intensity (0–100 scale) from RGB values:
            R={r}, G={g}, B={b}.
            Respond in JSON format:
            {{
              "intensity_score": <value>,
              "color_description": "<dominant color>",
              "analysis_notes": "<short explanation>"
            }}
            """
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(gemini_api_url, json=payload)
            if res.status_code == 200:
                result = res.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                st.success("AI Analysis Complete ✅")
                st.json(text)
            else:
                st.error(f"API Error: {res.status_code}")
    else:
        st.warning("Upload an image to enable AI prediction.")
