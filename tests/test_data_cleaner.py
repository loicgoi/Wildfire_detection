import pytest
import pandas as pd
import numpy as np
from prepare_data import data_cleaner as dc


@pytest.fixture(scope="module")
def sample_data():
    """Crée un dataset fictif pour les tests"""
    # Images DataFrame
    images_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "width": [100, 200, 150],
            "height": [100, 200, 150],
            "file_name": ["img1.jpg", "img2.jpg", "img3.jpg"],
        }
    )

    # Annotations DataFrame avec bbox au format [x, y, width, height]
    annotations_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "image_id": [1, 1, 2, 2, 3, 3],
            "bbox": [
                [10, 10, 50, 50],  # Valide
                [20, 20, 30, 30],  # Valide
                [50, 50, 100, 100],  # Valide
                [0, 0, 250, 250],  # Dépasse les dimensions de l'image 2
                [10, 10, 5, 20],  # Valide mais aire incohérente
                [10, 10, -5, 20],  # Largeur négative (invalide)
            ],
            "area": [2500, 900, 10000, 62500, 200, -100],  # Aire incohérente pour id=5
            "category_id": [1, 1, 2, 2, 1, 1],
        }
    )

    # Categories DataFrame
    categories_df = pd.DataFrame({"id": [1, 2], "name": ["cat1", "cat2"]})

    # Merged DataFrame
    merged_df = pd.merge(
        annotations_df,
        images_df,
        left_on="image_id",
        right_on="id",
        suffixes=("_annotation", "_image"),
    )

    return images_df, annotations_df, categories_df, merged_df


def test_expand_bbox(sample_data):
    """Teste l'expansion des bbox en colonnes séparées"""
    _, annotations_df, _, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df.copy())

    # Vérifie que toutes les colonnes sont présentes
    expected_cols = ["x", "y", "width", "height"]
    assert all(col in df_expanded.columns for col in expected_cols)

    # Vérifie les valeurs extraites
    assert df_expanded.iloc[0]["x"] == 10
    assert df_expanded.iloc[0]["y"] == 10
    assert df_expanded.iloc[0]["width"] == 50
    assert df_expanded.iloc[0]["height"] == 50


def test_detect_invalid_bbox_values(sample_data):
    """Teste la détection des bbox invalides"""
    _, annotations_df, _, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df.copy())
    invalid = dc.detect_invalid_bbox_values(df_expanded)

    # Vérifie que le résultat est un DataFrame
    assert isinstance(invalid, pd.DataFrame)

    # Vérifie qu'on détecte bien la bbox avec largeur négative
    assert len(invalid) == 1
    assert invalid.iloc[0]["id"] == 6  # ID de l'annotation invalide


def test_detect_inconsistent_area(sample_data):
    """Teste la détection des aires incohérentes"""
    _, annotations_df, _, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df.copy())
    inconsistent = dc.detect_inconsistent_area(df_expanded)

    # Vérifie que le résultat est un DataFrame
    assert isinstance(inconsistent, pd.DataFrame)

    # Vérifie qu'on détecte bien l'aire incorrecte
    assert len(inconsistent) == 1
    assert inconsistent.iloc[0]["id"] == 5  # ID de l'annotation avec aire incorrecte


def test_check_bbox_vs_image(sample_data):
    """Teste la détection des bbox dépassant les dimensions de l'image"""
    images_df, annotations_df, _, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df.copy())

    # Filtrer les bbox invalides avant de tester les dépassements
    df_valid = df_expanded[(df_expanded["width"] > 0) & (df_expanded["height"] > 0)]

    # Vérification manuelle pour débogage
    print("\nVérification manuelle des bbox:")
    for _, row in df_valid.iterrows():
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]
        # Récupérer les dimensions de l'image correspondante
        img_info = images_df[images_df["id"] == row["image_id"]]
        if not img_info.empty:
            img_w = img_info.iloc[0]["width"]
            img_h = img_info.iloc[0]["height"]
            out_of_bounds = (x < 0) or (y < 0) or (x + w > img_w) or (y + h > img_h)
            print(
                f"ID {row['id']}: bbox=({x},{y},{w},{h}), img=({img_w},{img_h}) -> {'HORS LIMITES' if out_of_bounds else 'OK'}"
            )

    out_of_bounds = dc.check_bbox_vs_image(df_valid, images_df)

    # Vérifie que le résultat est un DataFrame
    assert isinstance(out_of_bounds, pd.DataFrame)

    # Afficher les résultats pour débogage
    print(
        f"\nRésultats de la fonction: {len(out_of_bounds)} bbox hors limites détectées"
    )
    if len(out_of_bounds) > 0:
        print(out_of_bounds[["id", "x", "y", "width", "height"]])

    # Vérifie qu'on détecte bien la bbox hors limites
    assert len(out_of_bounds) == 1
    assert out_of_bounds.iloc[0]["id"] == 4  # ID de l'annotation hors limites
