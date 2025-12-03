import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
import requests

# -----------------------------------------------------
# 🧠 Title & Description
# -----------------------------------------------------
st.set_page_config(page_title="AI Fluorescence Analyzer", page_icon="💡", layout="wide")

st.title("💡 AI Fluorescence Analyzer for Fluoride Detection")
st.write("""
Upload a fluorescence image (e.g., from a smartphone or microscope).  
This app will:
1. Extract **RGB color values** from the image,  
2. Estimate **relative fluorescence intensity**, and  
3. (Optionally) use **AI (Gemini)** to interpret your result.
""")

# -----------------------------------------------------
# 🧪 Image Upload
# -----------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload Fluorescence Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fluorescence Image", use_container_width=True)

    # Convert image to NumPy array
    img_array = np.array(image)
    avg_rgb = img_array.mean(axis=(0, 1))
    r, g, b = avg_rgb

    st.subheader("🎨 Extracted Color Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Red (R)", f"{r:.1f}")
    col2.metric("Green (G)", f"{g:.1f}")
    col3.metric("Blue (B)", f"{b:.1f}")

    # -----------------------------------------------------
    # 💡 Simple Fluorescence Intensity Estimation
    # -----------------------------------------------------
    intensity = (r + g + b) / 3
    st.markdown(f"### 🔆 Estimated Relative Intensity: **{intensity:.1f} (0–255 scale)**")

    st.info("Tip: Higher RGB values indicate stronger fluorescence signal intensity.")

    # -----------------------------------------------------
    # 🤖 Optional AI-based Analysis (Gemini)
    # -----------------------------------------------------
    st.markdown("#### 🤖 Ask AI to Analyze Fluorescence")

    if st.button("Run AI Analysis (Gemini Model)"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            st.error("⚠️ Please add your Gemini API key in Streamlit Secrets (Settings → Secrets).")
            st.stop()

        # Prepare the request
       const geminiApiUrl = `https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
        prompt = f"""
        Analyze this fluorescence image used for fluoride detection.
        The extracted RGB values are R={r:.1f}, G={g:.1f}, B={b:.1f}.
        Estimate the relative fluorescence intensity on a scale of 0–100,
        and provide a short analytical comment in 2 sentences.
        """

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/jpeg", "data": base64_img}}
                ]
            }]
        }

        with st.spinner("Analyzing fluorescence with Gemini..."):
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    st.success("✅ AI Analysis Complete")
                    st.markdown("### 📘 Gemini AI Interpretation")
                    st.write(text)
                except Exception:
                    st.error("AI response received but could not be parsed. Check the console.")
                    st.json(data)
            else:
                st.error(f"API error: {response.status_code}")
                st.text(response.text)

else:
    st.info("👆 Upload an image to begin.")
