import torch
from model import get_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

test_dataset = datasets.ImageFolder(
    root="/Users/christianrafferty/Documents/pneumonia_recognition/data/test",
    transform=transform
)

print("Class mapping:", test_dataset.class_to_idx)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def test():
    model = get_model()
    model.load_state_dict(torch.load("/Users/christianrafferty/Documents/pneumonia_recognition/model.pth",map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")