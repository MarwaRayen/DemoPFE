from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from captum.attr import LayerGradCam
import matplotlib.cm as cm
from captum.attr import DeepLift
import numpy as np
from scipy.ndimage import gaussian_filter
from captum.attr import GuidedBackprop
from captum.attr import Occlusion
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Occlusion
from matplotlib.colors import LinearSegmentedColormap


def compute_integrated_gradients(model, clarus_tensor, plexelite_tensor, predicted_class, device):
    model.eval()

    # Move to device but DO NOT add .unsqueeze(0)
    clarus_tensor = clarus_tensor.to(device)
    plexelite_tensor = plexelite_tensor.to(device)

    def clarus_forward(x):
        plex_batch = torch.zeros(x.size(0), *plexelite_tensor.shape[1:], device=device)
        return model(x, plex_batch)

    def plexelite_forward(x):
        clarus_batch = torch.zeros(x.size(0), *clarus_tensor.shape[1:], device=device)
        return model(clarus_batch, x)

    clarus_ig = IntegratedGradients(clarus_forward)
    plexelite_ig = IntegratedGradients(plexelite_forward)

    clarus_attr = clarus_ig.attribute(
        clarus_tensor, 
        baselines=torch.zeros_like(clarus_tensor), 
        target=predicted_class,
        n_steps=50
    )
    plexelite_attr = plexelite_ig.attribute(
        plexelite_tensor, 
        baselines=torch.zeros_like(plexelite_tensor), 
        target=predicted_class,
        n_steps=50
    )

    return clarus_attr, plexelite_attr





def visualize_attributions(attr, input_tensor, title="Attribution Map"):
    attr = torch.abs(attr).squeeze().cpu().numpy()
    attr = np.mean(attr, axis=0)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(attr, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(title)
    return fig


def generate_ig_visualization_streamlit(clarus_tensor, plexelite_tensor, clarus_attr, plexelite_attr, predicted_label):
    def process_for_display(img_tensor, attr_tensor):
    

        attr = torch.abs(attr_tensor).squeeze().cpu().permute(1, 2, 0).numpy()
        img = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        
        attr = np.mean(attr, axis=-1)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        
        heatmap = attr ** 2
        threshold = np.percentile(heatmap, 75)
        heatmap = np.where(heatmap > threshold, heatmap, 0)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            heatmap = heatmap ** 0.3
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        cmap = plt.cm.get_cmap('jet')
        colored_heatmap = cmap(heatmap)[:, :, :3]

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        blended = np.uint8(img * 255 * 0.3 + colored_heatmap * 255 * 0.7)

        # Resize for less zoom effect
        blended = cv2.resize(blended, (512, 512), interpolation=cv2.INTER_AREA)
        return blended

    blended_clarus = process_for_display(clarus_tensor, clarus_attr)
    blended_plexelite = process_for_display(plexelite_tensor, plexelite_attr)

    # Plot and return figures
    fig1, ax1 = plt.subplots()
    ax1.imshow(blended_clarus)
    ax1.axis('off')
    ax1.set_title(f"CLARUS IG - Predicted Class {predicted_label}")

    fig2, ax2 = plt.subplots()
    ax2.imshow(blended_plexelite)
    ax2.axis('off')
    ax2.set_title(f"PLEXELITE IG - Predicted Class {predicted_label}")

    return fig1, fig2


def compute_gradcam(model, clarus_tensor, plexelite_tensor, predicted_class, device):
    model.eval()
    clarus_tensor = clarus_tensor.to(device).requires_grad_()
    plexelite_tensor = plexelite_tensor.to(device).requires_grad_()

    activations, gradients = {}, {}

    def save_act_grad(name):
        def forward_hook(module, input, output):
            activations[name] = output
        def backward_hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0]
        return forward_hook, backward_hook

    # Register hooks
    f_cl, b_cl = save_act_grad("clarus")
    f_pl, b_pl = save_act_grad("plexelite")

    h_cl_f = model.clarus_backbone.features[12].register_forward_hook(f_cl)
    h_cl_b = model.clarus_backbone.features[12].register_backward_hook(b_cl)
    h_pl_f = model.plexelite_backbone.features[12].register_forward_hook(f_pl)
    h_pl_b = model.plexelite_backbone.features[12].register_backward_hook(b_pl)

    # Forward + backward
    output = model(clarus_tensor, plexelite_tensor)
    model.zero_grad()
    output[0, predicted_class].backward()

    def generate_cam(acts, grads):
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1)).squeeze(0)
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    clarus_cam = generate_cam(activations["clarus"], gradients["clarus"])
    plexelite_cam = generate_cam(activations["plexelite"], gradients["plexelite"])

    # Cleanup
    h_cl_f.remove(); h_cl_b.remove()
    h_pl_f.remove(); h_pl_b.remove()

    return clarus_cam, plexelite_cam


def generate_gradcam_visualization_streamlit(img_tensor, cam, title):
    img = img_tensor.squeeze().cpu().permute(1, 2, 0).detach().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    blended = np.uint8(img * 255 * 0.3 + cam * 0.7)

    fig, ax = plt.subplots()
    ax.imshow(blended)
    ax.set_title(title)
    ax.axis('off')
    return fig



def generate_deeplift_visualization_streamlit(model, clarus_tensor, plexelite_tensor, predicted_class, device):
    

    class ClarusWrapper(torch.nn.Module):
        def __init__(self, model, plexelite_placeholder):
            super().__init__()
            self.model = model
            self.plexelite_placeholder = plexelite_placeholder

        def forward(self, x):
            return self.model(x, self.plexelite_placeholder.expand_as(x))

    class PlexeliteWrapper(torch.nn.Module):
        def __init__(self, model, clarus_placeholder):
            super().__init__()
            self.model = model
            self.clarus_placeholder = clarus_placeholder

        def forward(self, x):
            return self.model(self.clarus_placeholder.expand_as(x), x)

    if clarus_tensor.ndim == 3:
        clarus_tensor = clarus_tensor.unsqueeze(0)
    if plexelite_tensor.ndim == 3:
        plexelite_tensor = plexelite_tensor.unsqueeze(0)

    clarus_tensor = clarus_tensor.to(device)
    plexelite_tensor = plexelite_tensor.to(device)

    clarus_zeros = torch.zeros_like(plexelite_tensor).to(device)
    plexelite_zeros = torch.zeros_like(clarus_tensor).to(device)

    clarus_wrapper = ClarusWrapper(model, clarus_zeros).to(device)
    plexelite_wrapper = PlexeliteWrapper(model, clarus_tensor).to(device)

    baseline = 0.5
    clarus_base = torch.ones_like(clarus_tensor) * baseline
    plexelite_base = torch.ones_like(plexelite_tensor) * baseline

    # DeepLIFT
    clarus_attr = DeepLift(clarus_wrapper).attribute(clarus_tensor, clarus_base, target=predicted_class)
    plexelite_attr = DeepLift(plexelite_wrapper).attribute(plexelite_tensor, plexelite_base, target=predicted_class)

    def process(attr, img_tensor):
        attr = attr.squeeze().detach().cpu().numpy()
        attr = np.abs(attr).mean(0)
        attr = gaussian_filter(attr, sigma=2)
        vmax = np.percentile(attr, 99)
        attr = np.clip(attr, 0, vmax)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-10)
        attr[attr < 0.2] = 0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_np = (img_tensor.squeeze().detach() * std + mean).cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_np)
        axs[0].axis('off')
        axs[0].set_title("Original Image")
        im = axs[1].imshow(attr, cmap='jet')
        plt.colorbar(im, ax=axs[1])
        axs[1].axis('off')
        axs[1].set_title("DeepLIFT Attribution")
        axs[2].imshow(img_np)
        axs[2].imshow(attr, cmap='jet', alpha=0.5)
        axs[2].axis('off')
        axs[2].set_title("Overlay")
        plt.tight_layout()
        return fig

    fig_clarus = process(clarus_attr, clarus_tensor)
    fig_plexelite = process(plexelite_attr, plexelite_tensor)

    return fig_clarus, fig_plexelite



import torch
from captum.attr import GuidedBackprop
import matplotlib.pyplot as plt
import numpy as np

def generate_guided_backprop_visualization_streamlit(model, clarus_tensor, plexelite_tensor, predicted_class, device):
    model.eval()

    class ClarusWrapper(torch.nn.Module):
        def __init__(self, model, plex_shape):
            super().__init__()
            self.model = model
            self.plex_shape = plex_shape

        def forward(self, x):
            dummy_plex = torch.zeros((x.size(0), *self.plex_shape[1:]), device=x.device)
            return self.model(x, dummy_plex)

    class PlexWrapper(torch.nn.Module):
        def __init__(self, model, clarus_shape):
            super().__init__()
            self.model = model
            self.clarus_shape = clarus_shape

        def forward(self, x):
            dummy_clarus = torch.zeros((x.size(0), *self.clarus_shape[1:]), device=x.device)
            return self.model(dummy_clarus, x)

    clarus_tensor = clarus_tensor.to(device)
    plexelite_tensor = plexelite_tensor.to(device)

    clarus_attr = GuidedBackprop(ClarusWrapper(model, plexelite_tensor.shape)).attribute(clarus_tensor, target=predicted_class)
    plex_attr = GuidedBackprop(PlexWrapper(model, clarus_tensor.shape)).attribute(plexelite_tensor, target=predicted_class)

    clarus_attr = torch.clamp(clarus_attr, min=0)
    plex_attr = torch.clamp(plex_attr, min=0)

    def prepare_blended(attr, img_tensor, title):
        img = img_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        attr_np = attr.squeeze().cpu().permute(1, 2, 0).numpy()
        heatmap = attr_np.sum(axis=2)

        heatmap = heatmap**2
        threshold = np.percentile(heatmap, 50)
        heatmap = np.where(heatmap > threshold, heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            heatmap = heatmap**0.3

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        cmap = plt.cm.get_cmap('jet')
        colored_heatmap = cmap(heatmap)[:, :, :3]
        blended = np.uint8(img * 255 * 0.3 + colored_heatmap * 255 * 0.7)

        fig, ax = plt.subplots()
        ax.imshow(blended)
        ax.set_title(title)
        ax.axis('off')
        return fig

    fig_clarus = prepare_blended(clarus_attr, clarus_tensor, "CLARUS Guided Backprop")
    fig_plexelite = prepare_blended(plex_attr, plexelite_tensor, "PLEXELITE Guided Backprop")

    return fig_clarus, fig_plexelite



def generate_occlusion_visualization_streamlit(clarus_tensor, plexelite_tensor, model, predicted_class, device):
    """
    Generate Occlusion maps for both modalities using the 'WhiteRed' colormap from your original implementation.
    Returns: fig_clarus, fig_plexelite — matplotlib figures ready to be displayed in Streamlit.
    """
    model.eval()

    # Ensure batch size = 1
    clarus_tensor = clarus_tensor.unsqueeze(0) if clarus_tensor.dim() == 3 else clarus_tensor
    plexelite_tensor = plexelite_tensor.unsqueeze(0) if plexelite_tensor.dim() == 3 else plexelite_tensor

    # Forward functions
    def forward_clarus(x):
        return model(x, torch.zeros_like(plexelite_tensor).to(device))
    
    def forward_plexelite(x):
        return model(torch.zeros_like(clarus_tensor).to(device), x)

    # Occlusion setup
    occlusion = Occlusion(forward_func=forward_clarus)
    clarus_attr = occlusion.attribute(
        inputs=clarus_tensor.to(device),
        target=predicted_class,
        sliding_window_shapes=(3, 32, 32),
        strides=(1, 16, 16),
        baselines=0
    )

    occlusion = Occlusion(forward_func=forward_plexelite)
    plex_attr = occlusion.attribute(
        inputs=plexelite_tensor.to(device),
        target=predicted_class,
        sliding_window_shapes=(3, 32, 32),
        strides=(1, 16, 16),
        baselines=0
    )

    def denormalize(tensor):
        device = tensor.device  # Get the current device of the input tensor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        return tensor * std + mean

    def prepare_image_and_attr(img_tensor, attr_tensor):
        img_np = denormalize(img_tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        attr_np = attr_tensor.squeeze(0).cpu().numpy()
        attr_np = np.mean(attr_np, axis=0)
        attr_np = (attr_np - np.min(attr_np)) / (np.max(attr_np) - np.min(attr_np) + 1e-10)
        return img_np, attr_np

    clarus_img_np, clarus_attr_np = prepare_image_and_attr(clarus_tensor, clarus_attr)
    plex_img_np, plex_attr_np = prepare_image_and_attr(plexelite_tensor, plex_attr)

    # Define colormap: white to red
    white_red_cmap = LinearSegmentedColormap.from_list("WhiteRed", ["white", "red"])

    def make_overlay(image_np, attr_map, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_np)
        ax.imshow(attr_map, cmap=white_red_cmap, alpha=0.5, vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title)
        return fig

    fig_clarus = make_overlay(clarus_img_np, clarus_attr_np, "CLARUS Occlusion")
    fig_plexelite = make_overlay(plex_img_np, plex_attr_np, "PLEXELITE Occlusion")

    return fig_clarus, fig_plexelite
