import argparse
import json
import torch
import platform
from pathlib import Path
from PIL import Image
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import densenet201

def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predict(image_path, model_path):
    image_path = Path(image_path)
    model_path = Path(model_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model = densenet201(weights=None)
    model.classifier = nn.Linear(1920, 2)

    device = get_device()
    model = model.to(device)

    # Resolve model checkpoint: accept file, directory, or nested best_auc
    def _resolve_model_file(p: Path) -> Path:
        if p.exists() and p.is_file():
            return p
        # common locations
        candidates = [p / "model.pth", p / "best_auc" / "model.pth"]
        for c in candidates:
            if c.exists():
                return c
        # search for any .pth under the path
        if p.exists() and p.is_dir():
            matches = list(p.rglob("*.pth"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Model checkpoint not found under {p}")

    model_file = _resolve_model_file(model_path)

    checkpoint = torch.load(model_file, map_location=device)
    # Unwrap common checkpoint wrappers
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        prediction = probs.argmax(dim=1).item()
        confidence = probs[0, prediction].item()

    label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return label, float(confidence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray image")
    parser.add_argument("--image", "-i", required=True, help="Path to the input image")
    parser.add_argument(
        "--model",
        "-m",
        default="models/best_auc",
        help="Path to the model directory or checkpoint file",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    label, confidence = predict(args.image, args.model)
    if args.json:
        print(json.dumps({"label": label, "confidence": confidence}))
    else:
        print(f"Prediction: {label} (confidence={confidence:.6f})")
