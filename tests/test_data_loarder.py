import pytest
from prepare_data.data_loader import load_annotations_to_df
from pathlib import Path

JSON_PATH = Path("data/satellite_wildfire_detection/_annotations.coco.json")


def test_load_annotations_ti_df():
    images_df, annotations_df, categories_df, merged_df = load_annotations_to_df(
        JSON_PATH
    )

    # Vérifie que les DataFrames ne sont pas vides
    assert not images_df.empty
    assert not annotations_df.empty
    assert not categories_df.empty
    assert not merged_df.empty

    # Vérifie la présence des colonnes clés
    assert "id" in images_df.columns
    assert "image_id" in annotations_df.columns
    assert "category_id" in annotations_df.columns
