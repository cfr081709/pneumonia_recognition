from train import train
from eval import eval
from test import test

epochs = 200

def main():
    train(epochs=epochs)
    eval()
    test()

if __name__ == "__main__":
    main()