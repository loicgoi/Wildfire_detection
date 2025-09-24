# models/train.py
from model import CONFIGS, load_model


def train_model(cfg, data_yaml="data.yaml"):
    """Entraîne un modèle YOLO selon une config donnée"""
    model = load_model(cfg["model"])
    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=640,
        save_period=10,
    )
    return results


if __name__ == "__main__":
    # Exemple : entraîner le modèle nano
    cfg = CONFIGS["yv8n"]
    train_model(cfg)
