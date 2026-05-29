from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights

def get_model(num_classes = 2, pretrained=True):
    """
    Returns a denseNet201 model adapted for binary classification
    """

    if pretrained:
        model =  densenet201(weights=DenseNet201_Weights.DEFAULT)
    else:
        model = densenet201(weights=None)
        
    model.classifier = nn.Linear(1920, num_classes)
    
    return model