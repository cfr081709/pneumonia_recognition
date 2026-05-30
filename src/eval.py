import torch
import platform
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import densenet201
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, roc_curve, auc, recall_score, f1_score

def evaluate(save_file_path, model_path, data_dir, plot=False):

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

    eval_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    model = densenet201(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    if platform.system() == 'Darwin':
        device = torch.device("mps")
    else:
        device = torch.device("cuda")

    model.classifier = nn.Linear(1920, 2)

    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model = model.to(device)

    model.eval()

    predArray = []
    actualArray = []
    probabilityArray = []

    with torch.no_grad():

        for images, labels in eval_loader:

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

    precision = precision_score(
        actualArray,
        predArray,
        zero_division=0
    )

    recall = recall_score(
        actualArray,
        predArray,
        zero_division=0
    )

    f1 = f1_score(
        actualArray,
        predArray,
        zero_division=0
    )

    confusionMatrix = confusion_matrix(
        actualArray,
        predArray,
        labels=[0, 1]
    )

    fpr, tpr, _ = roc_curve(
        actualArray,
        probabilityArray
    )

    roc_auc = auc(fpr, tpr)

    results.append({
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

    print(f"""
        ===== Evaluation Results =====

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

    df = pd.DataFrame(results)
    df.to_csv(f"{save_file_path}/evaluation_results.csv", index=False)

    if plot:

        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.savefig(f"{save_file_path}/evaluation_plots.png")

        plt.show()