# Model Card

## Model Summary

This project adapts a pretrained DenseNet-201 model from `torchvision` for binary chest X-ray classification:

- Class 0: `NORMAL`
- Class 1: `PNEUMONIA`

The final classifier layer is replaced with a two-output linear layer, and the model is trained with cross-entropy loss.

## Intended Use

This model is intended for education, experimentation, and demonstrating a transfer-learning workflow in PyTorch. It can be used to explore:

- ImageFolder-based dataset loading
- Transfer learning with DenseNet
- Basic medical image preprocessing
- Train/validation/test classification evaluation

## Not Intended For

This model is not intended for:

- Clinical diagnosis
- Patient triage
- Treatment decisions
- Screening real patients
- Deployment in medical, hospital, or public-health workflows

## Inputs

The current preprocessing pipeline:

1. Resizes images.
2. Center-crops to 224 x 224.
3. Converts grayscale images to 3 channels.
4. Converts images to tensors.
5. Applies ImageNet normalization.

## Outputs

The model returns two logits. The predicted class is selected with `torch.argmax`.

## Limitations

- The validation set has only 16 images, so validation accuracy is not a reliable estimate of generalization.
- The dataset is imbalanced toward pneumonia in the training split.
- The code does not currently report sensitivity, specificity, precision, recall, F1 score, ROC-AUC, or a confusion matrix.
- The project does not include calibration analysis or confidence-threshold tuning.
- The current scripts use repository-relative paths from `src/paths.py`, but experiment settings are still embedded in code rather than external config.
- A high test score on this dataset may reflect dataset-specific artifacts rather than clinically robust pneumonia detection.

## Recommended Next Steps

- Add command-line arguments for alternate data directories and checkpoint paths.
- Add reproducible configuration for epochs, batch size, learning rate, and checkpoint paths.
- Track metrics beyond accuracy, especially recall/sensitivity for pneumonia and false-negative counts.
- Add a confusion matrix and per-class metrics.
- Save training logs and random seeds for reproducibility.
- Validate on an external dataset before making any broader claims.
