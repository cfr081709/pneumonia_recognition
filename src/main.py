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
    "model_path": "/Users/christianrafferty/pneumonia_recognition/models/pneumonia_model.pth",
    "test_save_path": "/Users/christianrafferty/pneumonia_recognition/results/test_results",
    "test_data_dir": "/Users/christianrafferty/pneumonia_recognition/data/test"
}

WINDOWS_CONFIG = {
    "epochs": 150,
    "training_results_path": "/Users/christianrafferty/pneumonia_recognition/results/train_results",
    "model_path": "/Users/christianrafferty/pneumonia_recognition/models/pneumonia_model.pth",
    "training_data_path": "/Users/christianrafferty/pneumonia_recognition/data/train",
    "plot": True,
    "evaluation_data_path": "/Users/christianrafferty/pneumonia_recognition/data/val",
    "evaluation_save_path": "/Users/christianrafferty/pneumonia_recognition/results/evaluation_results",
    "model_path": "/Users/christianrafferty/pneumonia_recognition/models/pneumonia_model.pth",
    "test_save_path": "/Users/christianrafferty/pneumonia_recognition/results/test_results",
    "test_data_dir": "/Users/christianrafferty/pneumonia_recognition/data/test"
}

def pipeline(config):
    train(epochs = config["epochs"],
          filepath = config["training_results_path"],
          model_path = config["model_path"],
          data_dir = config["training_data_path"],
          plot = config["plot"])
    evaluate(save_file_path=config["evaluation_save_path"], 
             model_path=config["model_path"], 
             data_dir=config["evaluation_data_path"], 
             plot=config["plot"])
    test(save_file_path=config["test_save_path"], 
         model_path=config["model_path"], 
         data_dir=config["test_data_dir"])
    
if __name__ == "__main__":
    if platform.system() == "Darwin":
        pipeline(MAC_CONFIG)
    else:
        pipeline(WINDOWS_CONFIG)