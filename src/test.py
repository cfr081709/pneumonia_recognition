import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights

data_dir = "/Users/christianrafferty/Documents/pneumonia_recognition/data/test"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

test_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,      # No shuffling for evaluation
    num_workers=4,
    pin_memory=True,
)

# Build model and load saved weights — don't retrain
model = densenet201(weights=None)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(1920, 2)

model_path = "/Users/christianrafferty/Documents/pneumonia_recognition/src/model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Predicted 0s (NORMAL):    {(preds == 0).sum().item()}")
    print(f"Predicted 1s (PNEUMONIA): {(preds == 1).sum().item()}")

if __name__ == '__main__':
    print(f"Evaluating on: {device}")
    print(f"Test samples: {len(dataset)}")
    evaluate()