import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# Shared Multimodal Model
# -------------------------------
class MultimodalModel(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(MultimodalModel, self).__init__()

        self.clarus_backbone = squeezenet1_1(pretrained=pretrained)
        self.clarus_backbone.classifier = nn.Identity()

        self.plexelite_backbone = squeezenet1_1(pretrained=pretrained)
        self.plexelite_backbone.classifier = nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x1, x2):
        x1 = self.global_pool(self.clarus_backbone.features(x1)).view(x1.size(0), -1)
        x2 = self.global_pool(self.plexelite_backbone.features(x2)).view(x2.size(0), -1)
        return self.fc(torch.cat((x1, x2), dim=1))


# -------------------------------
# Load Model
# -------------------------------
def load_multimodal_model(weights_path, device):
    model = MultimodalModel(num_classes=6)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# -------------------------------
# Preprocessing (for Classification)
# -------------------------------
def preprocess_images_classification(clarus_img, plexelite_img):
    transform_clarus = transforms.Compose([
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_plexelite = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    clarus_tensor = transform_clarus(clarus_img).unsqueeze(0)
    plexelite_tensor = transform_plexelite(plexelite_img).unsqueeze(0)

    return clarus_tensor, plexelite_tensor


# -------------------------------
# Preprocessing (for Prediction, with CLAHE)
# -------------------------------
def apply_clahe_to_grayscale_image(img):
    """Apply CLAHE to a grayscale image and convert to RGB."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    rgb_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb_img)


# def preprocess_images_prediction(clarus_img, plexelite_img_bmp):
#     plexelite_gray = np.array(plexelite_img_bmp.convert("L"))  # ensure grayscale
#     plexelite_clahe = apply_clahe_to_grayscale_image(plexelite_gray)

#     transform_clarus = transforms.Compose([
#         transforms.CenterCrop(448),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     transform_plexelite = transforms.Compose([
#         transforms.Resize((448, 448)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     clarus_tensor = transform_clarus(clarus_img).unsqueeze(0)
#     plexelite_tensor = transform_plexelite(plexelite_clahe).unsqueeze(0)

#     return clarus_tensor, plexelite_tensor


def preprocess_images_prediction(clarus_img, plexelite_img_bmp):
    plexelite_gray = np.array(plexelite_img_bmp.convert("L"))  # ensure grayscale
    plexelite_clahe = apply_clahe_to_grayscale_image(plexelite_gray)

    transform_clarus = transforms.Compose([
        transforms.Resize((512, 512)),  # instead of CenterCrop
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_plexelite = transforms.Compose([
        transforms.Resize((512, 512)),  # keep this for consistency
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    clarus_tensor = transform_clarus(clarus_img).unsqueeze(0)
    plexelite_tensor = transform_plexelite(plexelite_clahe).unsqueeze(0)

    return clarus_tensor, plexelite_tensor



# -------------------------------
# Inference
# -------------------------------
def predict_from_images(model, clarus_tensor, plexelite_tensor, device):
    clarus_tensor = clarus_tensor.to(device)
    plexelite_tensor = plexelite_tensor.to(device)

    with torch.no_grad():
        outputs = model(clarus_tensor, plexelite_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_class = int(probs.argmax())

    return predicted_class, probs
