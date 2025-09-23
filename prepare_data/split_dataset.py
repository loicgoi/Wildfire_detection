import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Chemins des dossiers
DATA_DIR = Path("data/satellite_wildfire_detection")
IMAGES_DIR = DATA_DIR  # tes images sont à la racine de ce dossier
ANNOTATIONS_FILE = DATA_DIR / "_annotations_cleaned.coco.json"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# Pourcentage d'entraînement
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


def split_dataset():
    # Charger le JSON
    with open(ANNOTATIONS_FILE, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    # Split
    train_images, val_images = train_test_split(
        images, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [
        ann for ann in annotations if ann["image_id"] in train_image_ids
    ]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    # Supprimer et recréer les dossiers s'ils existent
    for d in [TRAIN_DIR, VAL_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Copier les images
    for img in train_images:
        src = IMAGES_DIR / img["file_name"]
        dst = TRAIN_DIR / img["file_name"]
        shutil.copy(src, dst)

    for img in val_images:
        src = IMAGES_DIR / img["file_name"]
        dst = VAL_DIR / img["file_name"]
        shutil.copy(src, dst)

    # Créer les nouveaux JSON
    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"],
    }
    val_json = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"],
    }

    with open(DATA_DIR / "_annotations_train.coco.json", "w") as f:
        json.dump(train_json, f, indent=4)

    with open(DATA_DIR / "_annotations_val.coco.json", "w") as f:
        json.dump(val_json, f, indent=4)

    print(
        f"Split terminé : {len(train_images)} images en train, {len(val_images)} en val."
    )


# Permet d'exécuter directement le script
if __name__ == "__main__":
    split_dataset()