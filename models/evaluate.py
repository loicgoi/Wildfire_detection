# models/evaluate.py
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from IPython.display import Image, display

VAL_DIR = Path("runs/val")


def display_metrics_image(val_path):
    img_path = val_path / "results.png"
    if img_path.exists():
        display(Image(filename=str(img_path)))
    else:
        print(f"Pas de results.png trouvé pour {val_path}")


def get_last_trained_model():
    detect_dir = Path("runs/detect")
    runs = [d for d in detect_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError("Aucun entraînement trouvé dans runs/detect/")
    last_run = max(runs, key=lambda d: d.stat().st_mtime)
    return last_run.name


def evaluate_model(model_name=None, split="test", data_yaml="data.yaml"):
    # si aucun modèle précisé, prend le dernier entraîné
    if model_name is None:
        model_name = get_last_trained_model()

    # chemin du meilleur poids
    best_model_path = Path("runs/detect") / model_name / "weights" / "best.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Pas de poids trouvés pour {model_name}")

    # créer un dossier unique pour l'évaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = VAL_DIR / f"{model_name}_{timestamp}"
    eval_path.mkdir(parents=True, exist_ok=True)

    # charger le modèle et évaluer
    print(f"Évaluation avec {best_model_path}")
    model = YOLO(str(best_model_path))
    results = model.val(data=data_yaml, split=split, save=True, project=str(eval_path))

    # afficher les métriques
    display_metrics_image(eval_path)

    return results.results_dict
