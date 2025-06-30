import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

from multimodal_utils import (
    load_multimodal_model,
    preprocess_images_classification,
    predict_from_images
)


# Set Streamlit page config
st.set_page_config(page_title="Multimodal DR Classification", layout="centered")
st.title("🔬 Multimodal Diabetic Retinopathy Classification")

# Upload CLARUS and PLEXELITE images
clarus_file = st.file_uploader("📤 Upload CLARUS image", type=["png", "jpg", "jpeg"])
plexelite_file = st.file_uploader("📤 Upload PLEXELITE image", type=["png", "jpg", "jpeg", "bmp"])

if clarus_file and plexelite_file:
    col1, col2 = st.columns(2)
    with col1:
        clarus_img = Image.open(clarus_file).convert("RGB")
        st.image(clarus_img, caption="🖼️ CLARUS Image", use_container_width=True)
    with col2:
        plexelite_img = Image.open(plexelite_file).convert("RGB")
        st.image(plexelite_img, caption="🖼️ PLEXELITE Image", use_container_width=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_multimodal_model("models/best_multimodal_model_label_sqz_classification.pth", device)

    # Preprocess inputs
    clarus_tensor, plexelite_tensor = preprocess_images_classification(clarus_img, plexelite_img)

    # Predict
    with st.spinner("🔍 Classifying..."):
        predicted_class, probabilities = predict_from_images(model, clarus_tensor, plexelite_tensor, device)

    # Class labels
    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)', 'PPR (5)']

    # Show result
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted DR Class: **{class_names[predicted_class]}**")

    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob:.4f} ({prob*100:.2f}%)")

    # Optional: Show bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(probabilities)), probabilities)
    bars[predicted_class].set_color('red')
    ax.set_xlabel("DR Class")
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    ax.set_xticks(range(len(probabilities)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
