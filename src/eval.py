import os
import torch
import platform
import pandas as pd
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import densenet201
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, roc_curve, auc, recall_score, f1_score


def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tune_threshold(actual, probabilities):
    best_threshold = 0.5
    best_f1 = -1

    for threshold in [i / 100 for i in range(5, 96)]:
        preds = [1 if prob >= threshold else 0 for prob in probabilities]
        score = f1_score(actual, preds, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


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

    device = get_device()

    model.classifier = nn.Linear(1920, 2)

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

        for images, labels in eval_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            pneumonia_probs = probs[:, 1]

            actualArray.extend(labels.cpu().numpy())

            probabilityArray.extend(
                pneumonia_probs.cpu().numpy()
            )

    best_threshold, threshold_f1 = tune_threshold(actualArray, probabilityArray)
    predArray = [1 if prob >= best_threshold else 0 for prob in probabilityArray]

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
        "Threshold": best_threshold,
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

    os.makedirs(save_file_path, exist_ok=True)
    os.makedirs(fr"{model_path}/best_auc", exist_ok=True)

    pd.DataFrame([{
        "Threshold": best_threshold,
        "Validation F1": threshold_f1,
    }]).to_csv(fr"{model_path}/best_auc/threshold_metrics.csv", index=False)

    with open(fr"{model_path}/best_auc/threshold.txt", "w") as threshold_file:
        threshold_file.write(str(best_threshold))

    print(f"""
        ===== Evaluation Results =====
          
        Threshold: {best_threshold:.2f}
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
