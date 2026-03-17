import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


CLASSES = [
    "Los_Angeles",
    "San_Diego",
    "SLO",
    "Bakersfield",
    "Riverside",
    "Anaheim",
]


def build_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    return model


def predict(image_path):
    device = torch.device("cpu")

    model = build_model()
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    predictions = {}

    image_files = [
        f for f in os.listdir(image_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with torch.no_grad():
        for filename in image_files:
            filepath = os.path.join(image_path, filename)

            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            outputs = model(image)
            pred_idx = outputs.argmax(dim=1).item()
            pred_class = CLASSES[pred_idx]

            predictions[filename] = pred_class

    return predictions