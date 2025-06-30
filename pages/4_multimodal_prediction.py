import streamlit as st
from PIL import Image
from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch


from multimodal_utils import (
    load_multimodal_model,
    preprocess_images_prediction,
    predict_from_images
)


from explainability_utils import compute_integrated_gradients, visualize_attributions

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


    # Integrated Gradients
    # Integrated Gradients
    with st.expander("🟡 Integrated Gradients", expanded=True):
        st.markdown("✅ Ranked best for **CLARUS** images.")

        try:
            with st.spinner("Computing Integrated Gradients..."):
                model.eval()

                # Define modality-isolated forward functions with batch-safe dummy tensors
                def clarus_forward(x):
                    batch_size = x.size(0)
                    dummy_plex = torch.zeros(batch_size, *plexelite_tensor.shape[1:], device=device)
                    return model(x, dummy_plex)

                def plexelite_forward(x):
                    batch_size = x.size(0)
                    dummy_clarus = torch.zeros(batch_size, *clarus_tensor.shape[1:], device=device)
                    return model(dummy_clarus, x)

                # Initialize Integrated Gradients
                clarus_ig = IntegratedGradients(clarus_forward)
                plexelite_ig = IntegratedGradients(plexelite_forward)

                # Compute attributions with lower n_steps for performance
                clarus_attr = clarus_ig.attribute(
                    clarus_tensor.to(device),
                    baselines=torch.zeros_like(clarus_tensor).to(device),
                    target=predicted_class,
                    n_steps=10
                )

                plexelite_attr = plexelite_ig.attribute(
                    plexelite_tensor.to(device),
                    baselines=torch.zeros_like(plexelite_tensor).to(device),
                    target=predicted_class,
                    n_steps=10
                )

                # Improved attribution visualization
                def visualize(attr, original):
                    try:
                        attr = torch.abs(attr).squeeze().cpu().detach().numpy()
                        if attr.ndim == 3:
                            attr = np.mean(attr, axis=0)  # better contrast than sum
                        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

                        original = original.squeeze().cpu().detach().numpy()
                        if original.ndim == 3:
                            original = np.transpose(original, (1, 2, 0))
                        original = (original - original.min()) / (original.max() - original.min() + 1e-8)

                        # Resize attr map if needed
                        if attr.shape != original.shape[:2]:
                            from skimage.transform import resize
                            attr = resize(attr, original.shape[:2], anti_aliasing=True)

                        heatmap = np.stack([attr]*3, axis=-1)
                        overlay = 0.3 * original + 0.7 * heatmap

                        return np.clip(overlay, 0, 1)
                    except Exception as e:
                        st.warning(f"Visualization failed: {str(e)}")
                        return None

                # Generate overlays
                clarus_overlay = visualize(clarus_attr, clarus_tensor)
                plexelite_overlay = visualize(plexelite_attr, plexelite_tensor)

            # Display side-by-side overlays
            col1, col2 = st.columns(2)
            with col1:
                if clarus_overlay is not None:
                    st.image(clarus_overlay, caption="CLARUS - Integrated Gradients", use_column_width=True)
            with col2:
                if plexelite_overlay is not None:
                    st.image(plexelite_overlay, caption="PLEXELITE - Integrated Gradients", use_column_width=True)

        except Exception as e:
            st.error(f"Integrated Gradients failed: {str(e)}")


                


    # Grad-CAM
    with st.expander("🔴 Grad-CAM", expanded=False):
        st.markdown("✅ Most informative for **PLEXELITE** images.")
        # Implement Grad-CAM attribution and display
        st.empty()

    # DeepLIFT
    with st.expander("🟣 DeepLIFT", expanded=False):
        st.markdown("✅ Effective on both modalities.")
        # Implement DeepLIFT attribution and display
        st.empty()

    # Guided Backpropagation
    with st.expander("🔵 Guided Backpropagation", expanded=False):
        st.markdown("✅ Highlighted fine features in **PLEXELITE**.")
        # Implement Guided Backpropagation attribution and display
        st.empty()

    # Occlusion
    with st.expander("🟢 Occlusion", expanded=False):
        st.markdown("✅ More informative for **PLEXELITE** structures.")
        # Implement Occlusion attribution and display
        st.empty()