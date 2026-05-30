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
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, roc_curve, auc, recall_score, f1_score

def train(epochs, save_file_path, model_path, data_dir,  plot=False):
    
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

    if platform.system() == 'Darwin':
        device = torch.device("mps")
    else:
        device = torch.device("cuda")

    model.classifier = nn.Linear(1920, 2)

    if device == torch.device("cuda"):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    
    results = []
    accuracy_scores = []
    precision_scores = []
    true_neg_over = []
    false_pos_over = []
    true_pos_over = []
    false_neg_over = []
    recall_scores = []
    f1_scores = []

    os.makedirs(os.path.dirname(model_path), exist_ok=True)


    for epoch in range(epochs):

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

            pneumonia_probs = probs[:, 1]

            predArray.extend(preds.cpu().numpy())
            actualArray.extend(labels.cpu().numpy())
            probabilityArray.extend(pneumonia_probs.detach().cpu().numpy())

        accuracy = accuracy_score(actualArray, predArray)

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

        true_pos_over.append(confusionMatrix[1, 1])
        false_pos_over.append(confusionMatrix[0, 1])
        true_neg_over.append(confusionMatrix[0, 0])
        false_neg_over.append(confusionMatrix[1, 0])
        
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)

        fpr, tpr, _ = roc_curve(actualArray, probabilityArray)

        roc_auc = auc(fpr, tpr)

        end_time = time.time()

        total_time = end_time - start_time

        results.append({
            "Epoch": epoch + 1,
            "Loss": total_loss,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "True Positives": confusionMatrix[1, 1],
            "False Positives": confusionMatrix[0, 1],
            "True Negatives": confusionMatrix[0, 0],
            "False Negatives": confusionMatrix[1, 0],
            "Time": total_time
        })



        print(f"""
            ===== TRAINING ====
              
            Epoch {epoch+1}/{epochs}
            Loss: {total_loss:.4f}

            Accuracy: {accuracy:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            F1-Score: {f1:.4f}
            ROC-AUC: {roc_auc:.4f}

            True Positives: {confusionMatrix[1, 1]}
            False Positives: {confusionMatrix[0, 1]}
            True Negatives: {confusionMatrix[0, 0]}
            False Negatives: {confusionMatrix[1, 0]}

            Time: {total_time}
            """)

        df = pd.DataFrame(results)

        df.to_csv(f"{save_file_path}/train_results.csv", index=False)

        if roc_auc > best_auc:
            best_auc = roc_auc
            torch.save(model.state_dict(), model_path)
        if(epoch + 1) % 25 == 0:
            torch.save(model.save_dict(), f"{model_path}/checkpoints/model_epoch_{epoch}")


    if plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 4, 1)
        plt.plot(accuracy)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(2, 4, 2)
        plt.plot(precision)
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')

        plt.subplot(2, 4, 3)
        plt.plot(recall)
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')

        plt.subplot(2, 4, 4)
        plt.plot(f1)
        plt.title('F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')

        plt.subplot(2, 4, 5)
        plt.plot(true_pos_over)
        plt.title('True Positives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')

        plt.subplot(2, 4, 6)
        plt.plot(false_pos_over)
        plt.title('False Positives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')

        plt.subplot(2, 4, 7)
        plt.plot(true_neg_over)
        plt.title('True Negatives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')

        plt.subplot(2, 4, 8)
        plt.plot(false_neg_over)
        plt.title('False Negatives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')

        plt.tight_layout()

        plt.savefig(f"{save_file_path}/train_plots.png")
        
        plt.show()

    