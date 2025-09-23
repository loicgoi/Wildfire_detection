import fiftyone as fo
from pathlib import Path
import subprocess


dataset_dir = Path("data/satellite_wildfire_detection")
coco_json = Path("data/satellite_wildfire_detection/_annotations_cleaned.coco.json")

dataset = fo.Dataset.from_dir(
    dataset_dir=".",
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=coco_json,
    name="wildfire_dataset",
    data_path="data/satellite_wildfire_detection",
)

session = fo.launch_app(dataset, remote=True, port=5151)

subprocess.run(["cmd.exe", "/c", "start", session.url])

session.wait()
