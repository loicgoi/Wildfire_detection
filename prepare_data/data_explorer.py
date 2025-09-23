import pandas as pd
from prepare_data.data_loader import load_annotations_to_df
from .data_loader import load_annotations_to_df

# Retourne le nombre total d’images
def get_total_images(df_images: pd.DataFrame) -> int:
    return len(df_images)


# Retourne le nombre total d’annotations
def get_total_annotations(df_annotations: pd.DataFrame) -> int:
    return len(df_annotations)


# Retourne la liste des catégories
def get_categories(df_category: pd.DataFrame) -> list:
    return df_category["name"].tolist()


# Retourne le nombre d’images par catégorie
def get_images_per_category(
    df_annotations: pd.DataFrame, df_category: pd.DataFrame
) -> pd.Series:
    merged = df_annotations.merge(df_category, left_on="category_id", right_on="id")
    return merged.groupby("name")["image_id"].nunique()


# Statistiques sur le nombre d’annotations par image
def get_annotation_stats(df_annotations: pd.DataFrame) -> dict:
    ann_per_image = df_annotations.groupby("image_id").size()
    return {
        "min": int(ann_per_image.min()),
        "max": int(ann_per_image.max()),
        "mean": float(ann_per_image.mean()),
        "median": float(ann_per_image.median()),
    }


if __name__ == "__main__":
    images_df, annotations_df, categories_df, merged_df = load_annotations_to_df(
        "data/satellite_wildfire_detection/_annotations.coco.json"
    )

    print("Nombre total d’images:", get_total_images(images_df))
    print("Nombre total d’annotations:", get_total_annotations(annotations_df))
    print("Catégories:", get_categories(categories_df))
    print(
        "Nombre d’images par catégorie:\n",
        get_images_per_category(annotations_df, categories_df),
    )
    print("Statistiques annotations par image:", get_annotation_stats(annotations_df))

