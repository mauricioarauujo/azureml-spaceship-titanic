from typing import Dict
import pandas as pd
import numpy as np
from src.config import RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER
from src.utils import (
    create_age_cat_features,
    create_cabin_region,
    create_counts_from_cat_features,
    PROCESSING_PARAMS,
)


def preprocess_data(input_data: pd.DataFrame, params: Dict):
    """Process the raw data.

    Args:
        input_data (pd.DataFrame): raw data
        params (Dict): parameters
        parameters (Dict): parameters
    """
    processed_data = input_data.copy()
    processed_data[["Cabin_Deck", "Cabin_Num", "Cabin_Side"]] = processed_data[
        "Cabin"
    ].str.split("/", expand=True)
    processed_data = processed_data.astype({"Cabin_Num": "float64"})
    processed_data = create_cabin_region(processed_data)

    processed_data["Nickname"] = processed_data["Name"].str.split(" ").str[1]
    processed_data["Group"] = (
        processed_data["PassengerId"]
        .apply(lambda x: x.split("_")[0])
        .astype(int)
    )
    processed_data = processed_data.drop(columns=["Cabin", "Name"])

    processed_data = create_counts_from_cat_features(processed_data)
    processed_data = create_age_cat_features(processed_data)

    for feature in params["filter_cat_features"].keys():
        processed_data[feature] = np.where(
            processed_data[feature].isin(params["filter_cat_features"][feature]),
            "Other",
            processed_data[feature],
        )

    return processed_data

def main():
    # Read raw data
    raw_data = pd.read_csv(RAW_DATA_FOLDER / "train.csv")
    # Preprocess data
    processed_data = preprocess_data(raw_data, PROCESSING_PARAMS)
    # Save processed data
    processed_data.to_csv(PROCESSED_DATA_FOLDER / "processed_data.csv", index=False)

if __name__ == "__main__":
    main()
