import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet201

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

filenameP = "/Users/christianrafferty/Documents/pneumonia_recognition/data/val/PNEUMONIA"
filenameN = "/Users/christianrafferty/Documents/pneumonia_recognition/data/val/NORMAL"

model = densenet201(weights=None)
model.classifier = nn.Linear(1920, 2)

model.load_state_dict(torch.load(
    "/Users/christianrafferty/Documents/pneumonia_recognition/src/model.pth",
    map_location=device
))

model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def run_prediction(filename):
    image = Image.open(filename).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()

    return "normal" if pred == 0 else "pneumonia"


def evaluate():
    normal_images = [f for f in os.listdir(filenameN) if not f.startswith('.')]
    pneumonia_images = [f for f in os.listdir(filenameP) if not f.startswith('.')]

    correct = 0
    total = 0

    for image_name in normal_images:
        path = os.path.join(filenameN, image_name)
        pred = run_prediction(path)
        correct += (pred == "normal")
        total += 1
        print(f"Normal → {pred}")

    for image_name in pneumonia_images:
        path = os.path.join(filenameP, image_name)
        pred = run_prediction(path)
        correct += (pred == "pneumonia")
        total += 1
        print(f"Pneumonia → {pred}")

    print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.2f}%)")


if __name__ == '__main__':
    print(f"Evaluating on: {device}")
    print(f"Normal images:    {len([f for f in os.listdir(filenameN) if not f.startswith('.')])}")
    print(f"Pneumonia images: {len([f for f in os.listdir(filenameP) if not f.startswith('.')])}")
    print()
    evaluate()