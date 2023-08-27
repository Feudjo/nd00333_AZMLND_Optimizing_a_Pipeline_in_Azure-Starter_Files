# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The dataset contains data about direct marketing campaigns of a bank institution.
Our goal is to predict if a client will subscribe a term deposit.

The best performing model was a logistic regression model with a regularization strengh of 31.1 and max iteration of 100.

## Scikit-learn Pipeline
Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
The model is a logistic regression model.

The dataset is cleaned and splitted into train and test set.
Hyperdrive was used for hyperparameters tuning. This automates the search of the best parameters by running multiple iterations of the model based on the parameter grid provided. In our case the following values were used.
- C: uniform(0.1, 100)
- max_iter: choice(100, 400, 500, 800)

The uniform distribution allows for an exploration of a range ofegularization strengths and ensures the model is tested under different levels of regularization.
On the other hand using specific values for max_iter ensures the hyperparameter is tested with known, meaningful values. This could save computational resources.
## AutoML

The AutoMl process scaled the data with MaxAbsScaler.
The best performing model in this case was the LightGBM. This is an ensemble model.

## Pipeline comparison
Notice that the AutoMl process scaled the data. This is important to speed up the convergence of the algorithm.
The accuracy of the lr model is slightly better than that of LightGBM.

## Future work
It could be interesting to test the generalization ability of the models on other unseen datapoints.
