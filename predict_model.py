import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


from config import DEVICE, SAVE_PATH
from grass_utils import detect_grass_mask

# ----------------------------
# Preprocessing (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ----------------------------
# Model definition
# ----------------------------
def create_model():
    model = models.mobilenet_v2(pretrained=False)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, 2)
    return model

# ----------------------------
# Load model ONCE
# ----------------------------
model = create_model().to(DEVICE)
state = torch.load(SAVE_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ----------------------------
# Prediction function
# ----------------------------
def predict_from_image(image_bytes: bytes):
    # Decode image
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Invalid image")

    # Crop grass ROI
    mask = detect_grass_mask(img_bgr)
    ys, xs = np.where(mask > 0)

    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        pad = 5

        y0 = max(y0 - pad, 0)
        x0 = max(x0 - pad, 0)
        y1 = min(y1 + pad, img_bgr.shape[0] - 1)
        x1 = min(x1 + pad, img_bgr.shape[1] - 1)

        img_bgr = img_bgr[y0:y1+1, x0:x1+1]

    # Convert to PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Transform
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    label_map = {
    0: "No Trimming Needed",
    1: "Trimming Needed"
    }
    label = label_map[pred]
 

    return {
        "prediction": label,
        "confidence_trimmed": float(probs[0]),
        "confidence_grown": float(probs[1])
    }



