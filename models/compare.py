from ultralytics import YOLO
import os
import pandas as pd
import glob


def compare_models(data_yaml="data.yaml", save_dir="runs/comparisons"):
    results_summary = []

    # Création du dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)

    # Recherche automatique de tous les sous-dossiers runs/detect/*
    for run_dir in glob.glob("runs/detect/*"):
        name = os.path.basename(run_dir)
        best_weights = os.path.join(run_dir, "weights", "best.pt")

        if not os.path.exists(best_weights):
            print(f"Pas de poids trouvés pour {name}, saute...")
            continue

        print(f"\n=== Validation du modèle {name} ===")
        model = YOLO(best_weights)  # Charge les poids déjà entraînés
        metrics = model.val(
            data=data_yaml,
            project=save_dir,  # dossier global des comparaisons
            name=name,  # sous-dossier par modèle
        )

        results_summary.append(
            {
                "name": name,
                "metrics": metrics.results_dict,
            }
        )

    # 🔹 Création du tableau comparatif
    if results_summary:
        df = pd.DataFrame(
            [
                {
                    "Model": c["name"],
                    "Precision": c["metrics"].get("metrics/precision(B)", None),
                    "Recall": c["metrics"].get("metrics/recall(B)", None),
                    "mAP50": c["metrics"].get("metrics/mAP50(B)", None),
                    "mAP50-95": c["metrics"].get("metrics/mAP50-95(B)", None),
                    "Fitness": c["metrics"].get("fitness", None),
                }
                for c in results_summary
            ]
        )

        print("\n=== Résultats comparatifs ===")
        print(df)

        # Sauvegarde dans un CSV pour garder une trace
        csv_path = os.path.join(save_dir, "comparison_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nTableau sauvegardé dans : {csv_path}")

    return results_summary
