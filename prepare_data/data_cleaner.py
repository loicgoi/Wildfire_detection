import json
from pathlib import Path
import pandas as pd
from .data_loader import load_annotations_to_df


def open_json(json_path: Path) -> dict:
    """Lit un fichier JSON et retourne son contenu sous forme de dictionnaire."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: dict, json_out_path: Path):
    """Sauvegarde un dictionnaire Python dans un fichier JSON."""
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def remove_unannotated_images(json_path: Path, json_out_path: Path, annotations_df):
    """Supprime les images sans annotations et sauvegarde un nouveau JSON."""
    data = open_json(json_path)

    annotated_ids = set(annotations_df["image_id"].astype(int))
    data["images"] = [img for img in data["images"] if img["id"] in annotated_ids]
    data["annotations"] = [
        ann for ann in data["annotations"] if ann["image_id"] in annotated_ids
    ]

    save_json(data, json_out_path)
    print(f"JSON nettoyé sauvegardé : {json_out_path}")


def remove_files_without_annotations(
    images_dir: Path, annotations_df: pd.DataFrame, images_df: pd.DataFrame
):

    annotated_ids = set(annotations_df["image_id"].astype(int))
    images_without_annotations = images_df[~images_df["id"].isin(annotated_ids)]

    removed_count = 0
    for f_name in images_without_annotations["file_name"]:
        f_path = images_dir / f_name
        if f_path.exists():
            f_path.unlink()
            removed_count += 1

    print(f"{removed_count} images non annotées supprimées du dossier {images_dir}")


def expand_bbox(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes x, y, width, height à partir de bbox."""
    annotations_df = annotations_df.copy()
    annotations_df[["x", "y", "width", "height"]] = pd.DataFrame(
        annotations_df["bbox"].tolist(), index=annotations_df.index
    )
    return annotations_df


def detect_invalid_bbox_values(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les bbox avec coordonnées ou dimensions invalides."""
    return annotations_df[
        (annotations_df["x"] < 0)
        | (annotations_df["y"] < 0)
        | (annotations_df["width"] <= 0)
        | (annotations_df["height"] <= 0)
    ]


def detect_inconsistent_area(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie que area correspond bien à width * height."""
    annotations_df = annotations_df.copy()
    annotations_df["calc_area"] = annotations_df["width"] * annotations_df["height"]
    annotations_df["area_diff"] = annotations_df["calc_area"] - annotations_df["area"]
    return annotations_df[annotations_df["area_diff"].abs() > 1e-3]


def check_bbox_vs_image(
    annotations_df: pd.DataFrame, images_df: pd.DataFrame, epsilon: float = 1.0
):
    """
    Vérifie que les bbox ne sortent pas des limites de l'image.
    Tolère un petit dépassement (epsilon) dû aux arrondis flottants.
    """
    image_sizes = images_df.set_index("id")[["width", "height"]].to_dict("index")

    def is_invalid(row):
        img_w = image_sizes[row["image_id"]]["width"]
        img_h = image_sizes[row["image_id"]]["height"]
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]

        x2, y2 = min(x + w, img_w), min(y + h, img_h)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return True
        if (x + w) > img_w + epsilon or (y + h) > img_h + epsilon:

            return True
        return False

    annotations_df = annotations_df.copy()
    annotations_df["invalid_bbox"] = annotations_df.apply(is_invalid, axis=1)
    return annotations_df[annotations_df["invalid_bbox"]]

