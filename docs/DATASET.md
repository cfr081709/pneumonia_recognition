# Dataset Notes

## Source

This project uses the Kaggle dataset [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

- Kaggle owner: `paultimothymooney`
- Owner name: Paul Mooney
- Kaggle slug: `paultimothymooney/chest-xray-pneumonia`
- Local folder: `data/`

The dataset contains pediatric chest X-ray images organized into `NORMAL` and `PNEUMONIA` classes, with the original Kaggle train, validation, and test split structure preserved in this repository.

## Local Counts

| Split | Normal | Pneumonia | Total |
| --- | ---: | ---: | ---: |
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

## Licensing and Attribution

Check the Kaggle dataset page for the authoritative license and terms before redistributing the images or trained artifacts. The Kaggle page should be treated as the source of truth for dataset usage rights.

Recommended attribution:

> Chest X-Ray Images (Pneumonia), Kaggle dataset by Paul Mooney (`paultimothymooney`): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Data Caveats

- The validation split is very small, with only 16 images total.
- Class distribution is imbalanced, especially in the training split.
- Images may contain acquisition-site, preprocessing, scanner, or patient-population artifacts that a model can learn instead of clinically meaningful lung features.
- Dataset performance should not be interpreted as clinical performance.
