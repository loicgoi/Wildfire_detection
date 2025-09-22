from pathlib import Path
from pipeline import run_pipeline

if __name__ == "__main__":
    json_path = Path("data/satellite_wildfire_detection/_annotations.coco.json")
    json_out_path = Path(
        "data/satellite_wildfire_detection/_annotations_cleaned.coco.json"
    )

    run_pipeline(json_path, json_out_path)
