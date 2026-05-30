import os
import time
import torch
import platform
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import densenet201, DenseNet201_Weights
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


def train(epochs, save_file_path, model_path, data_dir, plot=False):

    best_auc = 0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
        pin_memory=False,
    )

    model = densenet201(weights=DenseNet201_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(1920, 2)

    device = torch.device("mps" if platform.system() == "Darwin" else "cuda")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

    results = []

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    true_pos_over = []
    false_pos_over = []
    true_neg_over = []
    false_neg_over = []

    os.makedirs(save_file_path, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        start_time = time.time()

        model.train()

        total_loss = 0

        predArray = []
        actualArray = []
        probabilityArray = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            predArray.extend(preds.detach().cpu().numpy())
            actualArray.extend(labels.detach().cpu().numpy())
            probabilityArray.extend(probs[:, 1].detach().cpu().numpy())

        accuracy = accuracy_score(actualArray, predArray)
        precision = precision_score(actualArray, predArray, zero_division=0)
        recall = recall_score(actualArray, predArray, zero_division=0)
        f1 = f1_score(actualArray, predArray, zero_division=0)

        cm = confusion_matrix(actualArray, predArray, labels=[0, 1])

        tp, fp, tn, fn = cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0]

        fpr, tpr, _ = roc_curve(actualArray, probabilityArray)
        roc_auc = auc(fpr, tpr)

        end_time = time.time()

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        true_pos_over.append(tp)
        false_pos_over.append(fp)
        true_neg_over.append(tn)
        false_neg_over.append(fn)

        results.append({
            "Epoch": epoch + 1,
            "Loss": total_loss,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Time": end_time - start_time
        })

        print(f"""
            Loss: {total_loss:.4f}
            Acc: {accuracy:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}
            AUC: {roc_auc:.4f}
            TP:{tp} FP:{fp} TN:{tn} FN:{fn}
            """)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_file_path, "train_results.csv"), index=False)

    if plot:
        
        print("accuracy_scores:", accuracy_scores)
        print("precision_scores:", precision_scores)
        print("recall_scores:", recall_scores)
        print("f1_scores:", f1_scores)

        if len(accuracy_scores) == 0:
            print("No data to plot")
            return

        epochs_range = range(1, len(accuracy_scores) + 1)

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 4, 1)
        plt.plot(epochs_range, accuracy_scores, marker='o')
        plt.title("Accuracy")

        plt.subplot(2, 4, 2)
        plt.plot(epochs_range, precision_scores, marker='o')
        plt.title("Precision")

        plt.subplot(2, 4, 3)
        plt.plot(epochs_range, recall_scores, marker='o')
        plt.title("Recall")

        plt.subplot(2, 4, 4)
        plt.plot(epochs_range, f1_scores, marker='o')
        plt.title("F1 Score")

        plt.subplot(2, 4, 5)
        plt.plot(epochs_range, true_pos_over, marker='o')
        plt.title("True Positives")

        plt.subplot(2, 4, 6)
        plt.plot(epochs_range, false_pos_over, marker='o')
        plt.title("False Positives")

        plt.subplot(2, 4, 7)
        plt.plot(epochs_range, true_neg_over, marker='o')
        plt.title("True Negatives")

        plt.subplot(2, 4, 8)
        plt.plot(epochs_range, false_neg_over, marker='o')
        plt.title("False Negatives")

        plt.tight_layout()
        plt.show()