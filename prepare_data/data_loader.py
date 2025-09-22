import json
import pandas as pd 

json_path = 'data/satellite_wildfire_detection/_annotations.coco.json'

def load_annotations_to_df(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    images_df = pd.DataFrame(data["images"])
    annotations_df = pd.DataFrame(data["annotations"])
    categories_df = pd.DataFrame(data["categories"])

    merged_df = annotations_df.merge(images_df, left_on="image_id", right_on="id", suffixes=("_ann", "_img"))
    merged_df = merged_df.merge(categories_df, left_on="category_id", right_on="id", suffixes=("", "_cat"))

    return images_df, annotations_df, categories_df, merged_df

load_annotations_to_df(json_path)

