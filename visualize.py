import fiftyone as fo
from pathlib import Path
import subprocess


def launch_fiftyone_dataset(
    dataset_dir: str,
    coco_json: str,
    dataset_name: str = "wildfire_dataset",
    port: int = 5151,
    remote: bool = True,
):
    """
    Charge un dataset COCO dans FiftyOne et lance l'interface.
    """
    dataset_dir = Path(dataset_dir)
    coco_json = Path(coco_json)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not coco_json.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_json}")

    # Charger le dataset COCO
    dataset = fo.Dataset.from_dir(
        dataset_dir=".",
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=coco_json,
        name=dataset_name,
        data_path=str(dataset_dir),
        overwrite=True,
    )

    # Lancer l'interface
    session = fo.launch_app(dataset, remote=remote, port=port)

    # Option Windows : ouvrir automatiquement le navigateur
    subprocess.run(["cmd.exe", "/c", "start", session.url])

    # Attendre la fermeture de la session
    session.wait()

    return dataset, session
