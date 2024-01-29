
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from azureml.core.run import Run
import sys

project_name = 'azureml-spaceship-titanic'
project_folder = os.getcwd().split(project_name)[0] + project_name
sys.path.append(project_folder)

from src.pipelines.preprocess import preprocess_data

# Get the experiment run context
run = Run.get_context()

# Parse the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength')
parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations for the solver')
args = parser.parse_args()

raw_df = pd.read_csv('../data/01_raw/train.csv')
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
