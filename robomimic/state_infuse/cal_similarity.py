import torch
import clip
from PIL import Image

def image_features(image,model,preprocess,device):
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features

