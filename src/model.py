import torch
import platform
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import densenet201

def predict(image_path, model_path):
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image = torchvision.io_read_image(image_path)

    model = densenet201(weights = None)

    model.classifier = nn.Linear(1920, 2)

    if platform.system() == "Darwin":
        if torch.mps.is_available():
            device = torch.device('mps')
    elif platform.system() != "Darwin":
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        raise Exception("GPU not availiable")

    model.load_state_dict(model_path)

    if torch.device() == "cuda":
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    model = model.to(device)

    model.eval()

    with torch.no_grad():
        
        image = image.to_device()
        
        output = model(image)

        probs = torch.softmax(output, dim=1)
        pneumonia_prob = probs[:, 1]
        normal_prob = probs[:, 0]
        preds = torch.argmax(probs, dim=1)

        if output == 1 and pneumonia_prob >= .75:
            print(f"Pneumonia | Conf: {pneumonia_prob}")
        elif output == 1 and pneumonia_prob < .75:
            print(f"Uncertain leans towards pneumonia | Conf: {pneumonia_prob}")
        elif output == 0 and normal_prob >= .75:
            print(f"Normal | Conf: {normal_prob}")
        elif output == 0 and normal_prob < .75:
            print(f"Uncertatin leans towards normal | Conf: {normal_prob}")