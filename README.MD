# Pneumonia Recognition

A PyTorch deep learning project that detects pneumonia in chest X-ray images using a fine-tuned DenseNet201 model.

## Overview

This project classifies chest X-rays as either:

- `NORMAL` / `0`
- `PNEUMONIA` / `1`

The model uses DenseNet201 transfer learning. Most pretrained layers remain frozen, while the final DenseNet block, final normalization layer, and classifier are fine-tuned for the pneumonia recognition task.

## Features

- DenseNet201 transfer learning
- Partial fine-tuning of `denseblock4`, `norm5`, and the classifier
- Class-weighted loss to reduce the effect of class imbalance
- CUDA, MPS, or CPU device selection
- Training, validation, and testing pipelines
- Threshold tuning on the validation set
- Accuracy, precision, recall, F1, ROC-AUC, and confusion matrix reporting
- ROC curve visualization

## Dataset

Dataset: Chest X-Ray Images (Pneumonia)

Classes:

- `NORMAL`
- `PNEUMONIA`

Images are resized to `224x224`, converted to three-channel grayscale, normalized with ImageNet statistics, and lightly augmented during training.

Training augmentations include:

- small rotations
- slight translation and scaling
- mild brightness and contrast jitter

**Chest X-Ray Images (Pneumonia)**  
Sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally published by Kermany et al.

> Kermany, D., Zhang, K., & Goldbaum, M. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*, 172(5), 1122-1131. https://doi.org/10.1016/j.cell.2018.02.010

## Installation

```bash
git clone https://github.com/cfr081709/pneumonia-recognition.git
cd pneumonia-recognition
pip install -r requirements.txt
```

## Training

```bash
python src/train.py
```

Training uses class weights computed from the training dataset. This helps compensate for the dataset imbalance between normal and pneumonia X-rays.

The model fine-tunes only the last part of DenseNet201:

- `features.denseblock4` at a low learning rate
- `features.norm5` at a low learning rate
- `classifier` at a higher learning rate

Training results are saved to:

```text
results/train_results/train_results.csv
results/train_results/train_plots.png
```

Training plot:

![Training plots](results/train_results/train_plots.png)

## Evaluation

```bash
python src/eval.py
```

Evaluation runs the model on the validation set and tunes the pneumonia decision threshold. Instead of always using `0.5`, the script searches thresholds from `0.05` to `0.95` and selects the threshold with the best validation F1 score.

Evaluation results are saved to:

```text
results/eval_results/evaluation_results.csv
results/eval_results/evaluation_plots.png
```

The selected threshold is saved to:

```text
models/best_auc/threshold.txt
models/best_auc/threshold_metrics.csv
```

## Testing

```bash
python src/test.py
```

Testing loads the saved threshold from evaluation when available. If no tuned threshold exists, it falls back to `0.5`.

Test results include:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- true positives
- false positives
- true negatives
- false negatives

Results are saved to:

```text
results/test_results/test_results.csv
```

## Full Pipeline

```bash
python src/main.py
```

The full pipeline runs training, evaluation, and testing in sequence.

## Project Structure

```text
pneumonia_recognition/
  data/
    train/
    val/
    test/
  models/
    best_auc/
  results/
    train_results/
    eval_results/
    test_results/
  src/
    train.py
    eval.py
    test.py
    main.py
    model.py
    data.py
```

## Technologies

- Python
- PyTorch
- Torchvision
- Pandas
- Matplotlib
- Scikit-learn
