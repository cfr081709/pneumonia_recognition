import __main__
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights

data_dir = "/Users/christianrafferty/Documents/pneumonia_recognition/data/train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = densenet201(weights=DenseNet201_Weights.DEFAULT)
model.classifier = nn.Linear(1920, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device=device)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

def train(epochs):
    print(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Prediction: {torch.argmax(outputs, dim=1).item()} Actual: {labels.item()}")
    torch.save(model.state_dict(), "/Users/christianrafferty/Documents/pneumonia_recognition/srcmodel.pth")
    print("Model saved as model.pth")

train(2)