import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet201, DenseNet201_Weights

data_dir = "/Users/christianrafferty/Documents/pneumonia_recognition/data/train"

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

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

model = densenet201(weights=DenseNet201_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(1920, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

def train(epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        last_outputs, last_labels = None, None

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_outputs, last_labels = outputs, labels

        pred = torch.argmax(last_outputs, dim=1)[0].item()
        actual = last_labels[0].item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - "
              f"Sample Prediction: {pred} | Actual: {actual}")
    save_path = "/Users/christianrafferty/Documents/pneumonia_recognition/src/model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    epochs = 25
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")
    print(f"Num epochs: {epochs}")
    print(f"Size of dataset {len(dataset)}")
    train(epochs)