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
The best model achieved an accuracy of 0.81. The parameters included using an VotingEnsemble model.

![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/3d01a82b-8df9-472f-a226-7c4c6837d01c)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/d7d50c32-17dc-487b-8ae9-5733763be523)


## Hyperparameter Tuning

### Model Choice
For hyperparameter tuning, I opted for the Logistic Regression model, specifically fine-tuning the regularization parameter (C) and maximum iterations (max_iter). Using Azure HyperDrive, I configured a Bandit Policy for early termination and Random Parameter Sampling to explore hyperparameter values. The objective was to maximize accuracy, allowing for a total of 20 runs and a maximum of 4 concurrent runs. This concise strategy aimed to enhance the model's performance efficiently.

### Results
The Logistic Regression model achieved an accuracy of 0.78 during hyperparameter tuning. The optimal hyperparameters for this performance were found to be C=1 and max_iter=200. To potentially enhance the model's performance further, additional exploration of hyperparameter combinations or alternative model architectures could be considered in future iterations.

![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/8c103922-cb8b-472b-a876-81ef99a673de)
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/ecd113e6-1a0f-432b-9204-707fd31a91c3)


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
The deployed model, based on AutoML, exhibits an accuracy higher than alternative models. To interact with the deployed service, follow the provided script below:
![image](https://github.com/mauricioarauujo/azureml-spaceship-titanic/assets/58861384/422ff993-0db6-4ba1-b56d-1c56d59538c4)



## Screen Recording
https://youtu.be/g4TafXmCfs0

