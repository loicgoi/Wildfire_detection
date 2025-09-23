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
TEST_DIR = DATA_DIR / "test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2  # sur l'ensemble total
TEST_RATIO = 0.1  # sur l'ensemble total
RANDOM_STATE = 42


def coco_to_yolo(annotations, images_dict):
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

    images_dict = {img["id"]: img for img in images}

    # Split train / temp (70% / 30%)
    train_images, temp_images = train_test_split(
        images, train_size=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    # Split temp en val / test (2/3 et 1/3 du temp pour obtenir 20%/10% total)
    val_images, test_images = train_test_split(
        temp_images,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        random_state=RANDOM_STATE,
    )

    # Récupérer les IDs
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    test_ids = {img["id"] for img in test_images}

    # Filtrer les annotations
    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids]
    test_annotations = [ann for ann in annotations if ann["image_id"] in test_ids]

    # Supprimer et recréer les dossiers
    for base_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if base_dir.exists():
            shutil.rmtree(base_dir)
        (base_dir / "images").mkdir(parents=True)
        (base_dir / "labels").mkdir(parents=True)

    # Copier les images
    for img in train_images:
        shutil.copy(
            IMAGES_DIR / img["file_name"], TRAIN_DIR / "images" / img["file_name"]
        )
    for img in val_images:
        shutil.copy(
            IMAGES_DIR / img["file_name"], VAL_DIR / "images" / img["file_name"]
        )
    for img in test_images:
        shutil.copy(
            IMAGES_DIR / img["file_name"], TEST_DIR / "images" / img["file_name"]
        )

    # Générer labels YOLO
    for split, annotations_split, dir_split in [
        ("train", train_annotations, TRAIN_DIR),
        ("val", val_annotations, VAL_DIR),
        ("test", test_annotations, TEST_DIR),
    ]:
        yolo_data = coco_to_yolo(annotations_split, images_dict)
        for f_name, lines in yolo_data.items():
            with open(dir_split / "labels" / (Path(f_name).stem + ".txt"), "w") as f:
                f.write("\n".join(lines))

    print(
        f"Split terminé : {len(train_images)} train, {len(val_images)} val, {len(test_images)} test."
    )


if __name__ == "__main__":
    split_dataset()