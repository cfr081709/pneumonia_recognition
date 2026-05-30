import os
import torch
import platform
from test import test
from train import train
from eval import evaluate

MAC_CONFIG = {
    "epochs": 150,
    "training_results_path": "/Users/christianrafferty/pneumonia_recognition/results/train_results",
    "model_path": "/Users/christianrafferty/pneumonia_recognition/models/pneumonia_model.pth",
    "training_data_path": "/Users/christianrafferty/pneumonia_recognition/data/train",
    "plot": True,
    "evaluation_data_path": "/Users/christianrafferty/pneumonia_recognition/data/val",
    "evaluation_save_path": "/Users/christianrafferty/pneumonia_recognition/results/evaluation_results",
    "test_save_path": "/Users/christianrafferty/pneumonia_recognition/results/test_results",
    "test_data_dir": "/Users/christianrafferty/pneumonia_recognition/data/test"
}

WINDOWS_CONFIG = {
    "epochs": 150,
    "training_results_path": r"C:\Users\Owner\pneumonia_recognition\results\train_results",
    "model_path": r"C:\Users\Owner\pneumonia_recognition\models",
    "training_data_path": r"C:\Users\Owner\pneumonia_recognition\data\train",
    "plot": True,
    "evaluation_data_path": r"C:\Users\Owner\pneumonia_recognition\data\val",
    "evaluation_save_path": r"C:\Users\Owner\pneumonia_recognition\results\eval_results",
    "test_save_path": r"C:\Users\Owner\pneumonia_recognition\results\test_results",
    "test_data_dir": r"C:\Users\Owner\pneumonia_recognition\data\test"
}

def pipeline(config):

    os.system('cls')

    print("\n\n === Beginning === \n\n")
    
    print("\n == Checking system == \n")

    if platform.system() == "Darwin":
        if not torch.mps.is_available() == False:
           raise Exception("GPU not availiable")
    elif torch.cuda.is_available() == False:
        raise Exception("GPU not availiable")

    print("\nGPU available continuing\n")

    print("\n == Beginning Training == \n")

    train(epochs = config["epochs"],
          save_file_path = config["training_results_path"],
          model_path = config["model_path"],
          data_dir = config["training_data_path"],
          plot = config["plot"])
    
    print("\n == Beginning Eval == \n")

    evaluate(save_file_path=config["evaluation_save_path"], 
             model_path=config["model_path"], 
             data_dir=config["evaluation_data_path"],)
    
    print("\n == Beginning Testing == \n")

    test(save_file_path=config["test_save_path"], 
         plot=config["plot"],
         model_path=config["model_path"],
         data_dir=config["test_data_dir"])
    
    print("\n == Finished == ")
    
if __name__ == "__main__":
    if platform.system() == "Darwin":
        pipeline(MAC_CONFIG)
    else:
        pipeline(WINDOWS_CONFIG)