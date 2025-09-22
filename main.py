from pathlib import Path
from pipeline import run_pipeline
from prepare_data.split_dataset import split_dataset

if __name__ == "__main__":
    json_path = Path("data/satellite_wildfire_detection/_annotations.coco.json")
    json_out_path = Path(
        "data/satellite_wildfire_detection/_annotations_cleaned.coco.json"
    )

    # On lance le pipeline de traintement
    run_pipeline(json_path, json_out_path)

    # On lance le split des données pour l'entrainement
    split_dataset()
