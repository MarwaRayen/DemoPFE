import streamlit as st
import sys
import os
from PIL import Image
import torch
import unimodal_utils as unimodal_utils
from unimodal_utils import load_model_DR_PRED, predict_from_image, load_model_DME_CLASS


# Set page config (optional)
st.set_page_config(page_title="Unimodal DR Prédiction", layout="centered")

# Title and description
st.title("🔬 Diabetic Retinopathy Prédiction")

# Image upload
uploaded_file = st.file_uploader("📤 Upload a retinal image (UWF)", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use the corrected model loading function
    model = load_model_DR_PRED(weights_path="models/uni_prediction_res50d_model.pth", device=device)
    
    # Predict
    with st.spinner("🔍 Classifying..."):
        predicted_class, probabilities = predict_from_image(image, model, device=device)

    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)', 'PPR (5)']
    
    # Show result
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted DR Class: **{class_names[predicted_class]}**")
    
    # Show full probabilities
    st.subheader("📊 Class Probabilities")
    
    
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Optional: Show probability bar chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(probabilities)), probabilities)
    ax.set_xlabel('DR Class')
    ax.set_ylabel('Probability')
    ax.set_title('Class Probabilities')
    ax.set_xticks(range(len(probabilities)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Highlight the predicted class
    bars[predicted_class].set_color('red')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.write("________________________________________________________________________")

    st.title("OMD Detection")
    
    
    # Use the corrected model loading function
    model = load_model_DME_CLASS(weights_path="models/best_res50d_model_DME_RC_classification.pth", device=device)
    
    # Predict
    with st.spinner("🔍 Classifying..."):
        predicted_class, probabilities = predict_from_image(image, model, device=device)

    class_names = ['Early (0)', 'Advanced (1)', 'Severe (2)']
    
    # Show result
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted DR Class: **{class_names[predicted_class]}**")