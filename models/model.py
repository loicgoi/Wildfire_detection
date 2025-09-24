from ultralytics import YOLO

# Liste de configurations disponibles
CONFIGS = {
    "yv8n": {"name": "yv8n", "model": "yolov8n.pt", "batch": 32, "epochs": 100},
    "yv8s": {"name": "yv8s", "model": "yolov8s.pt", "batch": 32, "epochs": 100},
    "y11n": {"name": "y11n", "model": "yolo11n.pt", "batch": 32, "epochs": 100},
}


def load_model(model_name):
    """Charge un modèle YOLO pré-entraîné"""
    return YOLO(model_name)
