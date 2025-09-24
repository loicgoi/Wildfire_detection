import argparse
from pathlib import Path
from pipeline import run_pipeline
from prepare_data.data_preparation import split_dataset
from visualize import launch_fiftyone_dataset
from models.model import CONFIGS
from models.train import train_model
from models.compare import compare_models
from models.evaluate import evaluate_model


def main(
    preprocessing=False, train=False, compare=False, evaluate=False, visualize=False
):
    json_path = Path("data/satellite_wildfire_detection/_annotations.coco.json")
    json_out_path = Path(
        "data/satellite_wildfire_detection/_annotations_cleaned.coco.json"
    )

    # Chemin pour visualize.py et FiftyOne
    dataset_dir = "data/satellite_wildfire_detection"
    coco_json = dataset_dir + "/_annotations_cleaned.coco.json"

    if preprocessing:
        # On lance le pipeline de traitement
        images_df, annotations_df, categories_df, merged_df = run_pipeline(
            json_path, json_out_path
        )

        # On lance le split des données pour l'entrainement
        split_dataset()

    if visualize:
        # Ouverture d'une fenêtre pour FiftyOne
        dataset, session = launch_fiftyone_dataset(dataset_dir, coco_json)

    if train:
        # Entraînement du modèle
        train_model(CONFIGS["yv8n"])

    if compare:
        # Comparaison de plusieurs modèles
        summary = compare_models(CONFIGS)
        print("Résumé des comparaisons :", summary)

    if evaluate:
        # Évaluation du meilleur modèle sur le test set
        model_name = "yv8n"
        results = evaluate_model(model_name, split="test")
        print("Résultats sur le test set :", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Wildfire Detection")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Import, traitement et split des données",
    )
    parser.add_argument("--train", action="store_true", help="Entrainer le modèle")
    parser.add_argument(
        "--compare", action="store_true", help="Comparer plusieurs modèles"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluer le meilleur modèle"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Lancer la visualisation FiftyOne"
    )

    args = parser.parse_args()

    main(
        preprocessing=args.preprocess,
        train=args.train,
        compare=args.compare,
        evaluate=args.evaluate,
        visualize=args.visualize,
    )
