
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from azureml.core import Run, Dataset

PROCESSING_PARAMS = {
    "filter_cat_features": {
        "Cabin_Deck": ["A", "D", "F", "T"],
        "Cabin_Region": ["G", "F"],
        "Destination": [None, "PSO J318.5-22"]
    }
}


def gen_counts_per_cat_col(
    df: pd.DataFrame, cat_col: str, feature_name: str
) -> pd.DataFrame:
    """Gera contagem por coluna categorica.

    Args:
        df (pd.DataFrame): dataset a ser modificado.
        cat_col (str): coluna categorial alvo.
        feature_name (str): nome final da feature.

    Returns:
        pd.DataFrame: dataset com a nova feature
    """

    df_count = df.groupby(cat_col).size().reset_index(name=feature_name)

    df = df.merge(df_count, on=[cat_col], how="left")

    return df


def create_counts_from_cat_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Cria features numericas a partir de categoricas."""
    cat_counts_params = {
        "Cabin_Num": {"Feature_Name": "People_in_Cabin_Num", "remove_col": True},
        "Cabin_Deck": {"Feature_Name": "People_in_Cabin_Deck", "remove_col": False},
        "Nickname": {"Feature_Name": "Family_Size", "remove_col": True},
        "Group": {"Feature_Name": "Group_Size", "remove_col": True},
    }

    df = input_df.copy()
    for col in list(cat_counts_params.keys()):

        df = gen_counts_per_cat_col(df, col, cat_counts_params[col]["Feature_Name"])
        if cat_counts_params[col]["remove_col"]:
            df = df.drop(columns=[col])

    return df


def create_age_cat_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas categoricas a partir da idade."""
    df = df_input.copy()
    df["Age_Cat"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 25, 50, 200],
        labels=["Child", "Teenager", "Pre_Adult", "Adult", "Elder"],
    )
    df = df.drop(columns=["Age"])

    return df


def create_expenditure_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria features de gasto.

    Args:
        df_input (pd.DataFrame): dataframe de entrada

    Returns:
        pd.DataFrame: dataframe com as novas features
    """
    df = df_input.copy()
    exp_feats = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["Expenditure"] = df[exp_feats].sum(axis=1)
    df["No_spending"] = (df["Expenditure"] == 0).astype(int)

    return df


def create_cabin_region(df_input: pd.DataFrame) -> pd.DataFrame:
    """Cria coluna de região do navio baseado na cabine.

    Args:
        df_input (pd.DataFrame): dataframe de entrada

    Returns:
        pd.DataFrame: dataframe com a nova feature
    """

    def _return_cabin_region(cabin_num: int) -> str:
        """Retorna a região da cabine."""
        if cabin_num < 300:
            return "A"
        elif cabin_num < 600:
            return "B"
        elif cabin_num < 900:
            return "C"
        elif cabin_num < 1200:
            return "D"
        elif cabin_num < 1500:
            return "E"
        elif cabin_num < 1800:
            return "F"
        else:
            return "G"

    df = df_input.copy()

    df["Cabin_Region"] = df["Cabin_Num"].apply(_return_cabin_region)

    return df


def preprocess_data(input_data: pd.DataFrame):
    """Process the raw data.

    Args:
        input_data (pd.DataFrame): raw data
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

    for feature in PROCESSING_PARAMS["filter_cat_features"].keys():
        processed_data[feature] = np.where(
            processed_data[feature].isin(PROCESSING_PARAMS["filter_cat_features"][feature]),
            "Other",
            processed_data[feature],
        )

    return processed_data


def main():
    # Get the experiment run context
    run = Run.get_context()

    # Parse the hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations for the solver')
    args = parser.parse_args()

    # Load the registered dataset
    dataset_name = 'Spaceship_Dataset'
    aml_dataset = Dataset.get_by_name(workspace=run.experiment.workspace, name=dataset_name)

    # Convert the dataset to a pandas DataFrame
    raw_df = aml_dataset.to_pandas_dataframe()
    print(raw_df.head())
    processed_df = preprocess_data(raw_df).drop(columns=["PassengerId"])
    X = processed_df.drop(columns=["Transported"])
    y = processed_df["Transported"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(C=args.C, max_iter=args.max_iter)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Log the hyperparameters and accuracy to the run
    run.log('C', args.C)
    run.log('max_iter', args.max_iter)
    run.log('Accuracy', accuracy)

    # Save the model
    model_filename = './hyperdrive_model.joblib'
    joblib.dump(value=model, filename=model_filename)

    # Upload the model file explicitly into the artifacts directory
    run.upload_file(name='outputs/' + model_filename, path_or_stream=model_filename)

    # Complete the run
    run.complete()


if __name__ == '__main__':
    main()
