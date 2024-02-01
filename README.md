# Spaceship Titanic Kaggle Competition in Azure ML

Kaggle's Spaceship-Titanic Project with AutoML and Hyperdrive tuning in Azure Machine Learning.

## Dataset

### Overview
The data was obtained from the Kaggle Spaceship-Titanic Competition. The dataset includes information about passengers aboard the Spaceship Titanic, such as age, class, fare paid, etc.

### Task
The task is to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

### Access
The data is stored in an Azure blob storage, and I am using the Azure Machine Learning dataset functionality to access it.

## Environment

### Local Environment
For local development and running, the required dependencies are specified in `notebooks/conda_dependencies.yml`. You can set up the environment using Conda with the following command:

```bash
conda env create --file notebooks/conda_dependencies.yml
conda activate sklearn-env
```
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/5cc178f7-43b4-4fcd-8fbd-5f046099c6c0)

### AzureML Workspace Environment
If you are working in an AzureML workspace, you can use the built-in "Python 3.8 - Azure ML" environment. Follow these steps:

- Select the "Python 3.8 - Azure ML" environment in your AzureML workspace.
- Authenticate to your AzureML workspace.
- Run the notebooks without any issues.


## Automated ML
 I used Azure AutoML with settings like experiment_timeout_minutes=30, max_concurrent_iterations=4, and primary_metric='accuracy'.
 
### Results
The best model achieved an accuracy of 0.81. The parameters included using an VotingEnsemble model.

![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/3d01a82b-8df9-472f-a226-7c4c6837d01c)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/327cb0ea-7f01-4b65-bb0c-571ba92027f7)


![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/e910104b-f03f-48c5-b600-d08cf8b532d1)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/38497f15-c5e6-4ab2-be0f-69a8aceddc7d)



## Hyperparameter Tuning

### Model Choice
For hyperparameter tuning, I opted for the Logistic Regression model, specifically fine-tuning the regularization parameter (C) and maximum iterations (max_iter). Using Azure HyperDrive, I configured a Bandit Policy for early termination and Random Parameter Sampling to explore hyperparameter values. The objective was to maximize accuracy, allowing for a total of 20 runs and a maximum of 4 concurrent runs. This concise strategy aimed to enhance the model's performance efficiently.

### Results
The Logistic Regression model achieved an accuracy of 0.78 during hyperparameter tuning. The optimal hyperparameters for this performance were found to be C=1 and max_iter=250. To potentially enhance the model's performance further, additional exploration of hyperparameter combinations or alternative model architectures could be considered in future iterations.

![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/a322b27c-212c-4857-bc28-b6e5bc479fd8)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/f66637ff-d998-45dd-bf0e-00482e36904c)


![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/1255a634-ba67-4d95-b281-9a014f02a899)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/535384d1-96e8-403a-840b-9cf7d05c295e)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/4ca8317d-12e1-4738-ad09-a52fee38509e)


## Model Deployment
The deployed model, based on AutoML, exhibits an accuracy higher than alternative models. 

To interact with the deployed service, follow the provided script below. You can use the following sample code to query the endpoint with a sample input:

```python
import json
import requests

scoring_uri = 'http://091674f3-7d52-4713-a563-29fdf9673dcd.westeurope.azurecontainer.io/score'

# If the service is authenticated, set the key or token
key = '2cloAPK95e4LlPLSINN0HXiMNWGOcJb5'

data = {
    "data": [
        {
            "HomePlanet": "Europa",
            "CryoSleep": "False",
            "Destination": "TRAPPIST-1e",
            "VIP": "False",
            "RoomService": 109.00,
            "FoodCourt": 1000,
            "ShoppingMall": 25.0,
            "Spa": 200.0,
            "VRDeck": 2.0,
            "Cabin_Deck": "B",
            "Cabin_Side": "P",
            "Cabin_Region": "A",
            "People_in_Cabin_Num": 14,
            "People_in_Cabin_Deck": 700,
            "Family_Size": 4,
            "Group_Size": 2,
            "Age_Cat": "Pre_Adult"
        }
    ],
    "method": "predict"
}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```


## Screen Recording
https://youtu.be/g4TafXmCfs0

## Future Improvements

To enhance and evolve this project in the future, consider the following areas for improvement:

### Model Enhancement
1. **Fine-tuning Parameters:** Experiment with different hyperparameters to optimize the model's performance further.
2. **Ensemble Techniques:** Explore ensemble learning methods to combine predictions from multiple models for improved accuracy.

### Platform and Deployment
1. **Scalability:** Evaluate the system's scalability to handle increased traffic or larger datasets.
2. **Continuous Integration/Continuous Deployment (CI/CD):** Implement CI/CD pipelines for streamlined model deployment and updates.

### Data
1. **Data Augmentation:** Investigate opportunities for data augmentation to increase the diversity of the training dataset.
2. **Feature Engineering:** Explore additional features or transformations to enhance the model's understanding of the data.

### Documentation and Monitoring
1. **Comprehensive Documentation:** Enhance project documentation to provide clearer instructions and explanations for users and contributors.
2. **Model Monitoring:** Implement a monitoring system to track the model's performance and detect deviations over time.

Feel free to contribute to these improvements or suggest additional enhancements to make this project even more robust and effective.

