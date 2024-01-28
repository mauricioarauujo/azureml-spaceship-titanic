import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    # Load the model into memory during initialization
    model_path = Model.get_model_path(model_name='AutoML_Best_Model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Convert the raw input data to a numpy array
        data = json.loads(raw_data)['data']
        input_data = np.array(data)

        # Perform inference using the loaded model
        result = model.predict(input_data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
