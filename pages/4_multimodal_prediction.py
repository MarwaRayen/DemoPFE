import streamlit as st
from PIL import Image
from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Occlusion



from multimodal_utils import (
    load_multimodal_model,
    preprocess_images_prediction,
    predict_from_images
)



from explainability_utils import (
    compute_integrated_gradients,
    generate_ig_visualization_streamlit,
    compute_gradcam,
    generate_gradcam_visualization_streamlit,
    generate_deeplift_visualization_streamlit,
    generate_guided_backprop_visualization_streamlit,
    generate_occlusion_visualization_streamlit
)


from explainability_utils import compute_integrated_gradients, visualize_attributions, generate_ig_visualization_streamlit

# Page configuration
st.set_page_config(page_title="Multimodal DR Prediction", layout="centered")
st.title("🔬 Multimodal Diabetic Retinopathy Prediction")

# Upload inputs
clarus_file = st.file_uploader("📤 Upload CLARUS image (UWF)", type=["png", "jpg", "jpeg"])
plexelite_file = st.file_uploader("📤 Upload PLEXELITE image (OCTA .bmp)", type=["bmp", "png", "jpg", "jpeg"])

# Proceed when both are uploaded
if clarus_file and plexelite_file:
    col1, col2 = st.columns(2)
    with col1:
        clarus_img = Image.open(clarus_file).convert("RGB")
        st.image(clarus_img, caption="🖼️ CLARUS Image", use_container_width=True)
    with col2:
        plexelite_img = Image.open(plexelite_file)
        st.image(plexelite_img, caption="🖼️ PLEXELITE Image", use_container_width=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_multimodal_model("models/best_multimodal_model_labelnext_sqzclahe_prediction.pth", device)

    # Preprocess (with CLAHE for PLEXELITE)
    clarus_tensor, plexelite_tensor = preprocess_images_prediction(clarus_img, plexelite_img)

    # Predict
    with st.spinner("🔍 Predicting future DR severity..."):
        predicted_class, probabilities = predict_from_images(model, clarus_tensor, plexelite_tensor, device)

    # Class labels
    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)', 'PPR (5)']

    # Show prediction
    st.subheader("✅ Prediction Result")
    st.success(f"Predicted Future DR Class: **{class_names[predicted_class]}**")

    # Show probabilities
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob:.4f} ({prob*100:.2f}%)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(probabilities)), probabilities)
    bars[predicted_class].set_color('red')
    ax.set_xlabel("DR Class")
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xticks(range(len(probabilities)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


    # Explainability Report Section
    st.markdown("---")
    st.subheader("🧠 Explainability Report")
    st.markdown("##### Modality Importance")
    st.markdown("Below are visualizations of the most influential regions in the input images using different explainability techniques. A note indicates which modality yielded stronger results.")


    with st.expander("🟡 Integrated Gradients", expanded=True):
        st.markdown("✅ Ranked best for **CLARUS** images.")
        
        with st.spinner("Generating Integrated Gradients explanations..."):

            # Compute IG attributions
            clarus_attr, plexelite_attr = compute_integrated_gradients(
                model, clarus_tensor, plexelite_tensor, predicted_class, device
            )

            # Enhanced visualization
            fig_clarus, fig_plexelite = generate_ig_visualization_streamlit(
                clarus_tensor, plexelite_tensor, clarus_attr, plexelite_attr, predicted_class
            )

            # Display
            st.pyplot(fig_clarus)
            st.pyplot(fig_plexelite)



    # Grad-CAM
    with st.expander("🔴 Grad-CAM", expanded=False):
        st.markdown("✅ Most informative for **PLEXELITE** images.")
        with st.spinner("Generating Grad-CAM explanations..."):
        
            clarus_cam, plexelite_cam = compute_gradcam(
                model, clarus_tensor, plexelite_tensor, predicted_class, device
            )

            fig_gc_clarus = generate_gradcam_visualization_streamlit(
                clarus_tensor, clarus_cam, f"CLARUS Grad-CAM - Class {predicted_class}"
            )
            fig_gc_plexelite = generate_gradcam_visualization_streamlit(
                plexelite_tensor, plexelite_cam, f"PLEXELITE Grad-CAM - Class {predicted_class}"
            )

            st.pyplot(fig_gc_clarus)
            st.pyplot(fig_gc_plexelite)



    # DeepLIFT
    with st.expander("🟣 DeepLIFT", expanded=False):
        st.markdown("✅ Effective on both modalities.")
        
        with st.spinner("Generating DeepLIFT explanations..."):

            fig_clarus_dl, fig_plexelite_dl = generate_deeplift_visualization_streamlit(
                model, clarus_tensor, plexelite_tensor, predicted_class, device
            )

            st.pyplot(fig_clarus_dl)
            st.pyplot(fig_plexelite_dl)
    # Guided Backpropagation
    with st.expander("🔵 Guided Backpropagation", expanded=False):
        st.markdown("✅ Highlighted fine features in **PLEXELITE**.")
        
        fig_clarus_gb, fig_plexelite_gb = generate_guided_backprop_visualization_streamlit(
            model, clarus_tensor, plexelite_tensor, predicted_class, device
        )

        st.pyplot(fig_clarus_gb)
        st.pyplot(fig_plexelite_gb)


    with st.expander("🟢 Occlusion", expanded=False):
        st.markdown("✅ More informative for **PLEXELITE** structures.")
        with st.spinner("Generating Occlusion explanations..."):
            fig_clarus, fig_plexelite = generate_occlusion_visualization_streamlit(
                clarus_tensor, plexelite_tensor, model, predicted_class, device
            )
            st.pyplot(fig_clarus)
            st.pyplot(fig_plexelite)

