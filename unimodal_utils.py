import torch
import timm
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch.nn as nn


# Resize + pad image to 512x512
def resize_image_pil(img, output_size=512):
    w, h = img.size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_height = int(round(output_size / aspect_ratio))
        new_width = output_size
    else:
        new_width = int(round(output_size * aspect_ratio))
        new_height = output_size

    img = img.resize((new_width, new_height), Image.LANCZOS)
    delta_w = output_size - new_width
    delta_h = output_size - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding)



# Standard transform (CenterCrop + Normalize)
def get_transform():
    return transforms.Compose([
        transforms.CenterCrop(400),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



# Use the EXACT same model class as in training
class RES50(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(RES50, self).__init__()
        
        # Load the ResNet50 model from timm
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)
    

class RES50D(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(RES50D, self).__init__()
        
        # Load the ResNet50D model from timm
        self.backbone = timm.create_model(
            'resnet50d',
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.backbone(x)
    


def load_model_DR_CLASS(weights_path, device='cpu'):
    """
    Load the SimpleModel with the given weights.
    """
    model = RES50(num_classes=6)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_model_DR_PRED(weights_path, device='cpu'):
    """
    Load the SimpleModel with the given weights.
    """
    model = RES50D(num_classes=6)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_from_image(image_pil, model, device='cpu'):
    """
    Preprocess an image and perform prediction with the given model.
    """
    image = resize_image_pil(image_pil)
    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_class = int(torch.argmax(output, dim=1).item())
    
    return predicted_class, probs



