import pytest
import pandas as pd
from prepare_data import data_cleaner as dc
from prepare_data.data_loader import load_annotations_to_df


@pytest.fixture(scope="module")
def load_dataset():
    json_path = "data/satellite_wildfire_detection/_annotations.coco.json"
    # renvoie les 4 DataFrames : images, annotations, categories, merged
    return load_annotations_to_df(json_path)


def test_expand_bbox(load_dataset):
    _, annotations_df, _, _ = load_dataset
    annotations_df = annotations_df.head(5)  # on teste sur un petit échantillon
    df_expanded = dc.expand_bbox(annotations_df)
    for col in ["x", "y", "width", "height"]:
        assert col in df_expanded.columns


def test_detect_invalid_bbox_values(load_dataset):
    _, annotations_df, _, _ = load_dataset
    annotations_df = dc.expand_bbox(annotations_df)
    invalid = dc.detect_invalid_bbox_values(annotations_df)
    assert isinstance(invalid, pd.DataFrame)


def test_detect_inconsistent_area(load_dataset):
    _, annotations_df, _, _ = load_dataset
    annotations_df = dc.expand_bbox(annotations_df)
    inconsistent = dc.detect_inconsistent_area(annotations_df)
    assert isinstance(inconsistent, pd.DataFrame)


def test_check_bbox_vs_image(load_dataset):
    images_df, annotations_df, _, _ = load_dataset
    annotations_df = dc.expand_bbox(annotations_df)
    out_of_bounds = dc.check_bbox_vs_image(annotations_df, images_df)
    assert isinstance(out_of_bounds, pd.DataFrame)
