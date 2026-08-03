import os
import torch
import platform
from pathlib import Path
from test import test
from train import train
from eval import evaluate

BASE_DIR = Path(__file__).resolve().parents[1]

COMMON_CONFIG = {
    "epochs": 150,
    "training_results_path": BASE_DIR / "results" / "train_results",
    "model_path": BASE_DIR / "models",
    "training_data_path": BASE_DIR / "data" / "train",
    "plot": True,
    "evaluation_data_path": BASE_DIR / "data" / "val",
    "evaluation_save_path": BASE_DIR / "results" / "evaluation_results",
    "test_save_path": BASE_DIR / "results" / "test_results",
    "test_data_dir": BASE_DIR / "data" / "test",
}

def pipeline(config):
    clear_cmd = "cls" if os.name == "nt" else "clear"
    os.system(clear_cmd)

    print("\n\n === Beginning === \n\n")
    print("\n == Checking system == \n")

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        print("Using MPS device")
    elif torch.cuda.is_available():
        print("Using CUDA device")
    else:
        print("Using CPU device")

    print("\n == Beginning Training == \n")

    train(
        epochs=config["epochs"],
        save_file_path=config["training_results_path"],
        model_path=config["model_path"],
        data_dir=config["training_data_path"],
        plot=config["plot"],
    )
    
    print("\n == Beginning Eval == \n")

    evaluate(
        save_file_path=config["evaluation_save_path"],
        model_path=config["model_path"],
        data_dir=config["evaluation_data_path"],
    )
    
    print("\n == Beginning Testing == \n")

    test(
        save_file_path=config["test_save_path"],
        model_path=config["model_path"],
        data_dir=config["test_data_dir"],
    )
    
    print("\n == Finished == ")

def runTest(config):
    print("\n == Beginning Testing == \n")

    test(
        save_file_path=config["test_save_path"],
        model_path=config["model_path"],
        data_dir=config["test_data_dir"],
    )

if __name__ == "__main__":
    pipeline(COMMON_CONFIG)
