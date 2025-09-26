import json
import pytest
from prepare_data.data_loader import load_annotations_to_df


@pytest.fixture
def fake_json(tmp_path):
    """Crée un JSON COCO minimal pour tester le loader."""
    data = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "img1.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "category_id": 1,
            }
        ],
        "categories": [{"id": 1, "name": "fire"}],
    }
    json_path = tmp_path / "annotations.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def test_load_annotations_to_df(fake_json):
    images_df, annotations_df, categories_df, merged_df = load_annotations_to_df(
        fake_json
    )

    # Vérifie que les DataFrames ne sont pas vides
    assert not images_df.empty
    assert not annotations_df.empty
    assert not categories_df.empty
    assert not merged_df.empty

    # Vérifie les colonnes essentielles
    assert {"id", "file_name", "width", "height"} <= set(images_df.columns)
    assert {"id", "image_id", "category_id"} <= set(annotations_df.columns)
    assert "name" in categories_df.columns
