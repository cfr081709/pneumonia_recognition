import shutil
from pathlib import Path


def organize_data(eval_path, train_path):
    eval_path = Path(eval_path)
    train_path = Path(train_path)

    normal_eval = eval_path / "NORMAL"
    pneumonia_eval = eval_path / "PNEUMONIA"
    normal_train = train_path / "NORMAL"
    pneumonia_train = train_path / "PNEUMONIA"

    normal_eval.mkdir(parents=True, exist_ok=True)
    pneumonia_eval.mkdir(parents=True, exist_ok=True)
    normal_train.mkdir(parents=True, exist_ok=True)
    pneumonia_train.mkdir(parents=True, exist_ok=True)

    count_p = sum(1 for _ in pneumonia_train.iterdir())
    count_n = sum(1 for _ in normal_train.iterdir())

    num_eval_p = int(0.15 * count_p)
    num_eval_n = int(0.15 * count_n)

    for file_name in list(normal_train.iterdir()):
        if sum(1 for _ in normal_eval.iterdir()) < num_eval_n:
            shutil.move(str(file_name), str(normal_eval))

    for file_name in list(pneumonia_train.iterdir()):
        if sum(1 for _ in pneumonia_eval.iterdir()) < num_eval_p:
            shutil.move(str(file_name), str(pneumonia_eval))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    organize_data(repo_root / "data" / "val", repo_root / "data" / "train")
