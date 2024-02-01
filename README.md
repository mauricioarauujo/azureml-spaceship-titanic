*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Spaceship Titanic Kaggle Competition in Azure ML

Kaggle's Spaceship-Titanic Project with AutoML and Hyperdrive tuning in Azure Machine Learning.


## Dataset

### Overview
The data was obtained from the Kaggle Spaceship-Titanic Competition. The dataset includes information about passengers aboard the Spaceship Titanic, such as age, class, fare paid, etc.

### Task
The task is to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

### Access
The data is stored in an Azure blob storage, and I am using the Azure Machine Learning dataset functionality to access it.


## Automated ML
 I used Azure AutoML with settings like experiment_timeout_minutes=30, max_concurrent_iterations=4, and primary_metric='accuracy'.
 

### Results
The best model achieved an accuracy of 0.85. The parameters included using an XGBoost model with a learning rate of 0.1 and max_depth of 5. I could have improved it by tuning more hyperparameters.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
