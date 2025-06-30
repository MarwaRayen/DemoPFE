import streamlit as st

st.set_page_config(page_title="AI Diagnostic App", layout="wide")

st.title("👁️ AI-Powered Retinal Diagnosis")
st.markdown("""
Welcome to the demo of our multimodal and unimodal deep learning models for **Diabetic Retinopathy (DR)** and **Macular Edema (OMD)** diagnosis.

Use the sidebar to navigate through:
            
- 🧠 **Unimodal Classification**
            
- 📈 **Unimodal Prediction**
            
- 🔄 **Multimodal Classification**
            
- 🧬 **Multimodal Prediction**

Each module allows you to upload images and view predictions, along with explainability visualizations.
""")