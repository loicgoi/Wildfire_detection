from ultralytics import YOLO
from pathlib import Path


def evaluate_model(model_name, split="test", data_yaml="data.yaml"):
    model_path = Path(f"runs/detect/{model_name}/weights/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pas de poids trouvés pour {model_name} ({model_path})"
        )

    model = YOLO(str(model_path))
    results = model.val(data=data_yaml, split=split)
    return results
