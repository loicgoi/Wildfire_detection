from pathlib import Path
from prepare_data.data_loader import load_annotations_to_df
from prepare_data.data_cleaner import (
    expand_bbox,
    detect_invalid_bbox_values,
    detect_inconsistent_area,
    check_bbox_vs_image,
    remove_unannotated_images,
    remove_files_without_annotations,
)


def run_pipeline(json_path: Path, json_out_path: Path):
    """
    Exécute le pipeline complet :
    1. Charger les données
    2. Étendre les bbox
    3. Détecter anomalies
    4. Nettoyer le JSON
    """
    # Charger les données
    images_df, annotations_df, categories_df, merged_df = load_annotations_to_df(
        json_path
    )

    # Etendre la bbox en 4 colonnes (x, y, width, heigth)
    annotations_df = expand_bbox(annotations_df)

    # Détecter les anomalies
    invalid_bbox_df = detect_invalid_bbox_values(annotations_df)
    inconsistent_area_df = detect_inconsistent_area(annotations_df)
    bbox_out_image_df = check_bbox_vs_image(annotations_df, images_df)

    print(f"Annotations invalide : {len(invalid_bbox_df)}")
    print(f"Annotations avec area incohérentes : {len(inconsistent_area_df)}")
    print(f"Annotations hors limites de l'image : {len(bbox_out_image_df)}")

    # Nettoyer le JSON
    remove_unannotated_images(json_path, json_out_path, annotations_df)

    # Supprimer les images sans annotations du disque
    images_dir = Path("data/satellite_wildfire_detection")
    remove_files_without_annotations(images_dir, annotations_df, images_df)

    return images_df, annotations_df, categories_df, merged_df
