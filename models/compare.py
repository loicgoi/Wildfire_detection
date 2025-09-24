from ultralytics import YOLO
from .model import CONFIGS
import os


def compare_models(configs, data_yaml="data.yaml"):
    results_summary = []

    for name, cfg in configs.items():
        # On suppose que chaque modèle est entraîné et stocké dans runs/detect/{name}
        run_dir = f"runs/detect/{name}"
        best_weights = os.path.join(run_dir, "weights", "best.pt")

        if not os.path.exists(best_weights):
            print(f"Pas de poids trouvés pour {name}, entraîne-le d'abord avec --train")
            continue

        print(f"\n=== Validation du modèle {name} ({cfg['model']}) ===")
        model = YOLO(best_weights)  # Charge les poids déjà entraînés
        metrics = model.val(data=data_yaml)  # Évalue sur le set de validation

        results_summary.append(
            {
                "name": name,
                "model": cfg["model"],
                "batch": cfg["batch"],
                "epochs": cfg["epochs"],
                "metrics": metrics.results_dict,
            }
        )

    return results_summary
