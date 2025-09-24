from ultralytics import YOLO


def evaluate_model(model_path, data_yaml="data.yaml", split="test"):
    """
    Évalue un modèle YOLO sur un split (val ou test).

    Args:
        model_path (str): chemin vers le modèle entraîné (.pt dans runs/detect/...)
        data_yaml (str): chemin vers le fichier YAML du dataset
        split (str): "val" ou "test"
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split)
    return results


if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"
    results = evaluate_model(model_path, split="test")
    print(results)
