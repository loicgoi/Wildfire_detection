import pytest
import pandas as pd
import prepare_data.data_cleaner as dc


@pytest.fixture
def sample_data():
    """Crée un dataset fictif pour tester les fonctions du cleaner."""
    images_df = pd.DataFrame(
        {
            "id": [1, 2],
            "width": [100, 200],
            "height": [100, 200],
            "file_name": ["img1.jpg", "img2.jpg"],
        }
    )
    annotations_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "image_id": [1, 1, 2, 2],
            "bbox": [
                [10, 10, 50, 50],  # valide
                [20, 20, -10, 30],  # largeur négative
                [0, 0, 200, 200],  # dépasse un peu les limites
                [5, 5, 10, 10],  # aire incohérente volontaire
            ],
            "area": [2500, -300, 40000, 200],  # volontairement incohérent pour id=4
            "category_id": [1, 1, 2, 2],
        }
    )
    categories_df = pd.DataFrame({"id": [1, 2], "name": ["cat1", "cat2"]})
    return images_df, annotations_df, categories_df


def test_expand_bbox(sample_data):
    _, annotations_df, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df)
    assert all(col in df_expanded.columns for col in ["x", "y", "width", "height"])
    assert df_expanded.iloc[0]["x"] == 10
    assert df_expanded.iloc[1]["width"] == -10


def test_detect_invalid_bbox_values(sample_data):
    _, annotations_df, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df)
    invalid = dc.detect_invalid_bbox_values(df_expanded)
    assert not invalid.empty
    assert invalid.iloc[0]["id"] == 2  # largeur négative


def test_detect_inconsistent_area(sample_data):
    _, annotations_df, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df)
    inconsistent = dc.detect_inconsistent_area(df_expanded)
    assert not inconsistent.empty
    assert 4 in inconsistent["id"].tolist()


def test_check_bbox_vs_image(sample_data):
    images_df, annotations_df, _ = sample_data
    df_expanded = dc.expand_bbox(annotations_df)
    out_of_bounds = dc.check_bbox_vs_image(df_expanded, images_df, epsilon=0.0)
    assert isinstance(out_of_bounds, pd.DataFrame)
    assert 2 in out_of_bounds["id"].tolist()  # largeur négative détectée


def test_remove_files_without_annotations(sample_data, tmp_path):
    images_df, annotations_df, _ = sample_data

    # Crée des fichiers factices
    for fname in images_df["file_name"]:
        (tmp_path / fname).write_text("fake")

    # Supprime fichiers sans annotations
    dc.remove_files_without_annotations(tmp_path, annotations_df, images_df)

    annotated_ids = set(annotations_df["image_id"])
    kept_files = images_df[images_df["id"].isin(annotated_ids)]["file_name"].tolist()
    for fname in kept_files:
        assert (tmp_path / fname).exists()
