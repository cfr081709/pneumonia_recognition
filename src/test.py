import torch
import platform
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms


def test(save_file_path, model_path, data_dir):

    results = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    test_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    model = densenet201(weights=None)

    model.classifier = nn.Linear(1920, 2)

    if platform.system() == "Darwin":
        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    if torch.device == "cuda":
        if torch.cuda.count() > 1:
            model = torch.nn.DataParallel(model)

    model = model.to(device)

    model.eval()

    predArray = []
    actualArray = []
    probabilityArray = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            pneumonia_probs = probs[:, 1]

            predArray.extend(preds.cpu().numpy())
            actualArray.extend(labels.cpu().numpy())

            probabilityArray.extend(
                pneumonia_probs.cpu().numpy()
            )

    accuracy = accuracy_score(
        actualArray,
        predArray
    )

    results.append({
        "Accuracy": accuracy
    })

    pd.DataFrame(results).to_csv(f"{save_file_path}/test_results.csv", index=False)

    print(f"Test Accuracy: {accuracy:.4f}")