import pytest
import pandas as pd
from prepare_data.data_explorer import (
    get_total_images,
    get_total_annotations,
    get_categories,
    get_images_per_category,
    get_annotation_stats,
)


@pytest.fixture
def dataset():
    """Jeu de données factice pour explorer les fonctions."""
    images_df = pd.DataFrame(
        {
            "id": [1, 2],
            "file_name": ["img1.jpg", "img2.jpg"],
            "width": [100, 200],
            "height": [100, 200],
        }
    )
    annotations_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "image_id": [1, 1, 2],
            "category_id": [1, 2, 1],
        }
    )
    categories_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["cat1", "cat2"],
        }
    )
    return images_df, annotations_df, categories_df


@pytest.mark.parametrize(
    "func, expected",
    [
        (get_total_images, 2),
        (get_total_annotations, 3),
    ],
)
def test_counts(func, dataset, expected):
    images_df, annotations_df, _ = dataset
    df = images_df if func == get_total_images else annotations_df
    assert func(df) == expected


def test_get_categories(dataset):
    _, _, categories_df = dataset
    assert get_categories(categories_df) == ["cat1", "cat2"]


def test_get_images_per_category(dataset):
    _, annotations_df, categories_df = dataset
    result = get_images_per_category(annotations_df, categories_df)
    assert isinstance(result, pd.Series)
    assert result["cat1"] == 2  # deux images avec la catégorie cat1
    assert result["cat2"] == 1


def test_get_annotation_stats(dataset):
    _, annotations_df, _ = dataset
    stats = get_annotation_stats(annotations_df)
    assert set(stats.keys()) == {"min", "max", "mean", "median"}
    assert stats["min"] >= 1
