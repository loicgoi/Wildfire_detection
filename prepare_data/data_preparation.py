import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Chemins des dossiers
DATA_DIR = Path("data/satellite_wildfire_detection")
IMAGES_DIR = DATA_DIR
ANNOTATIONS_FILE = DATA_DIR / "_annotations_cleaned.coco.json"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

TRAIN_RATIO = 0.8
RANDOM_STATE = 42


def coco_to_yolo(annotations, images_dict):
    """
    Convertit les annotations COCO en fichiers YOLO.
    annotations: liste d'annotations
    images_dict: dict {image_id: image_info}
    """
    yolo_data = {}
    for ann in annotations:
        img_id = ann["image_id"]
        img_info = images_dict[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        line = f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        f_name = img_info["file_name"]
        yolo_data.setdefault(f_name, []).append(line)

    return yolo_data


def split_dataset():
    # Charger le JSON nettoyé
    with open(ANNOTATIONS_FILE, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    # Dictionnaire pour accès rapide aux infos images
    images_dict = {img["id"]: img for img in images}

    # Split train/val
    train_images, val_images = train_test_split(
        images, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [
        ann for ann in annotations if ann["image_id"] in train_image_ids
    ]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    # Supprimer entièrement les dossiers existants
    for base_dir in [TRAIN_DIR, VAL_DIR]:
        if base_dir.exists():
            shutil.rmtree(base_dir)
        (base_dir / "images").mkdir(parents=True)
        (base_dir / "labels").mkdir(parents=True)

    # Copier les images annotées
    for img in train_images:
        shutil.copy(
            IMAGES_DIR / img["file_name"], TRAIN_DIR / "images" / img["file_name"]
        )
    for img in val_images:
        shutil.copy(
            IMAGES_DIR / img["file_name"], VAL_DIR / "images" / img["file_name"]
        )

    # Convertir COCO → YOLO et générer fichiers labels
    train_yolo = coco_to_yolo(train_annotations, images_dict)
    val_yolo = coco_to_yolo(val_annotations, images_dict)

    for f_name, lines in train_yolo.items():
        with open(TRAIN_DIR / "labels" / (Path(f_name).stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    for f_name, lines in val_yolo.items():
        with open(VAL_DIR / "labels" / (Path(f_name).stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

    print(
        f"Split terminé : {len(train_images)} images en train, {len(val_images)} en val."
    )


if __name__ == "__main__":
    split_dataset()
