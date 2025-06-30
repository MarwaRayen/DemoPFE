# explainability_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, clarus_tensor, plexelite_tensor, device, predicted_label):
    model.eval()

    def clarus_forward(x):
        return model(x, torch.zeros((x.shape[0], *plexelite_tensor.shape[1:]), device=device))

    def plexelite_forward(x):
        return model(torch.zeros((x.shape[0], *clarus_tensor.shape[1:]), device=device), x)

    clarus_ig = IntegratedGradients(clarus_forward)
    plexelite_ig = IntegratedGradients(plexelite_forward)

    clarus_attr = clarus_ig.attribute(
        clarus_tensor.to(device),
        baselines=torch.zeros_like(clarus_tensor).to(device),
        target=predicted_label,
        n_steps=50
    )

    plexelite_attr = plexelite_ig.attribute(
        plexelite_tensor.to(device),
        baselines=torch.zeros_like(plexelite_tensor).to(device),
        target=predicted_label,
        n_steps=50
    )

    return clarus_attr, plexelite_attr


def visualize_attributions(input_tensor, attr_tensor, title):
    """
    Overlay IG heatmap on input image.
    """
    input_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    attr_np = attr_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

    # Normalize input
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

    # Attribution: absolute sum across channels
    heatmap = np.abs(attr_np).sum(axis=-1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Enhance contrast
    heatmap = heatmap ** 2
    threshold = np.percentile(heatmap, 75)
    heatmap = np.where(heatmap > threshold, heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        heatmap = heatmap ** 0.3

    colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]
    blended = 0.3 * input_np + 0.7 * colored_heatmap
    blended = np.clip(blended, 0, 1)

    # Plot with matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(blended)
    ax.set_title(title)
    ax.axis('off')
    return fig
