import pytest
import pandas as pd
from prepare_data.data_loader import load_annotations_to_df
from prepare_data.data_explorer import (
    get_total_images,
    get_total_annotations,
    get_categories,
    get_images_per_category,
    get_annotation_stats,
)


@pytest.fixture(scope="module")
def dataset():
    json_path = "data/satellite_wildfire_detection/_annotations.coco.json"
    images_df, annotations_df, categories_df, _ = load_annotations_to_df(json_path)
    return images_df, annotations_df, categories_df


def test_get_total_images(dataset):
    images_df, _, _ = dataset
    total = get_total_images(images_df)
    assert total > 0


def test_get_total_annotations(dataset):
    _, annotations_df, _ = dataset
    total = get_total_annotations(annotations_df)
    assert total > 0


def test_get_categories(dataset):
    _, _, categories_df = dataset
    categories = get_categories(categories_df)
    assert len(categories) > 0


def test_get_images_per_category(dataset):
    _, annotations_df, categories_df = dataset
    result = get_images_per_category(annotations_df, categories_df)
    assert isinstance(result, pd.Series)
    assert not result.empty


def test_get_annotation_stats(dataset):
    _, annotations_df, _ = dataset
    stats = get_annotation_stats(annotations_df)
    expected_keys = {"min", "max", "mean", "median"}
    assert expected_keys <= stats.keys()
