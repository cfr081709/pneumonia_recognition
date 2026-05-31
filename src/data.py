import shutil
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

def organizeData(evalPath, trainPath):
    countP = (sum(1 for _ in Path(f"{evalPath}/PNEUMONIA").iterdir()) +
              sum(1 for _ in Path(f"{trainPath}/PNEUMONIA").iterdir()))
    countN = (sum(1 for _ in Path(f"{evalPath}/NORMAL").iterdir()) +
              sum(1 for _ in Path(f"{trainPath}/NORMAL").iterdir()))

    numEvalP = int(0.15 * countP)
    numEvalN = int(0.15 * countN)

    for file_name in Path(f"{trainPath}/NORMAL").iterdir():
        if sum(1 for _ in Path(f"{evalPath}/NORMAL").iterdir()) < numEvalN:
            shutil.move(file_name, f"{evalPath}/NORMAL")

    for file_name in Path(f"{trainPath}/PNEUMONIA").iterdir():
        if sum(1 for _ in Path(f"{evalPath}/PNEUMONIA").iterdir()) < numEvalP:
            shutil.move(file_name, f"{evalPath}/PNEUMONIA")

if __name__ == "__main__":
    organizeData(r"C:\Users\Owner\pneumonia_recognition\data\val", r"C:\Users\Owner\pneumonia_recognition\data\train")