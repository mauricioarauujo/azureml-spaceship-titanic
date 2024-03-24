
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score
from azureml.core import Run, Dataset

from scipy.stats import randint


def main():
    # Get the experiment run context
    run = Run.get_context()
    parser = argparse.ArgumentParser()

    # Input dataset
    parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

    # Parse the hyperparameters
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations for the solver')
    args = parser.parse_args()

    # Load the registered dataset
    processed_df = run.input_datasets['training_data'].to_pandas_dataframe() 

    X = processed_df.drop(columns=["Transported"])
    y = processed_df["Transported"]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Train a logistic regression model
    numeric_transformer = Pipeline(
        steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=args.C, max_iter=args.max_iter))
        ]
    )
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)

    # Log the hyperparameters and accuracy to the run
    run.log('C', args.C)
    run.log('max_iter', args.max_iter)
    run.log('Accuracy', accuracy)

    # Save the model in the run outputs
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/hyperdrive_model.joblib')

    # Complete the run
    run.complete()


if __name__ == '__main__':
    main()
