# models/train.py
from .model import CONFIGS, load_model


def train_model(cfg, data_yaml="data.yaml"):
    """Entraîne un modèle YOLO selon une config donnée"""
    model = load_model(cfg["model"])
    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=640,
        save=True,
        save_period=10,
        project="runs/detect",
        name=cfg["name"],
    )
    return results
