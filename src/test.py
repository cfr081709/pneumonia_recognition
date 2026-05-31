import torch
import platform
import pandas as pd
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import densenet201
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from torchvision import datasets, transforms


def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_threshold(model_path):
    threshold_path = Path(model_path) / "best_auc" / "threshold.txt"

    if not threshold_path.exists():
        return 0.5

    return float(threshold_path.read_text().strip())


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

    device = get_device()

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    try:
        model.load_state_dict(torch.load(fr"{model_path}/best_auc/model.pth"))
    except:
        print("\nNo best_auc model.pth | If no checkpoint model.pth program will quit\n")

        best_score = 0

        best_file = None

        checkpoint_files = list(Path(fr"{model_path}/checkpoints").iterdir())
        print("Checkpoint dir contents:", checkpoint_files)

        for file in Path(fr"{model_path}/checkpoints").iterdir():
            if file.suffix != ".csv":
                continue
            df = pd.read_csv(file)
            score_column = "score" if "score" in df.columns else "best_auc"
            score = df[score_column].iloc[0]
            if score > best_score:
                best_score = score
                best_file = file.with_suffix(".pth")
        
        try:
            model.load_state_dict(torch.load(best_file))
        except:
            print("\nCould not load checkpoint file | Exiting\n")
            raise Exception("\nNo model.pth files loaded\n")

    model = model.to(device)

    model.eval()

    actualArray = []
    probabilityArray = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            pneumonia_probs = probs[:, 1]

            actualArray.extend(labels.cpu().numpy())

            probabilityArray.extend(
                pneumonia_probs.cpu().numpy()
            )

    threshold = load_threshold(model_path)
    predArray = [1 if prob >= threshold else 0 for prob in probabilityArray]

    accuracy = accuracy_score(
        actualArray,
        predArray
    )

    precision = precision_score(actualArray, predArray, zero_division=0)
    recall = recall_score(actualArray, predArray, zero_division=0)
    f1 = f1_score(actualArray, predArray, zero_division=0)

    confusionMatrix = confusion_matrix(actualArray, predArray, labels=[0, 1])

    fpr, tpr, _ = roc_curve(actualArray, probabilityArray)
    roc_auc = auc(fpr, tpr)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "True Positives": confusionMatrix[1, 1],
        "False Positives": confusionMatrix[0, 1],
        "True Negatives": confusionMatrix[0, 0],
        "False Negatives": confusionMatrix[1, 0]
    })

    pd.DataFrame(results).to_csv(f"{save_file_path}/test_results.csv", index=False)

    print(f"""
        ===== Test Results =====

        Threshold: {threshold:.2f}
        Accuracy: {accuracy:.4f}
        Precision: {precision:.4f}
        Recall: {recall:.4f}
        F1-Score: {f1:.4f}
        ROC-AUC: {roc_auc:.4f}

        True Positives: {confusionMatrix[1, 1]}
        False Positives: {confusionMatrix[0, 1]}
        True Negatives: {confusionMatrix[0, 0]}
        False Negatives: {confusionMatrix[1, 0]}
        """)
