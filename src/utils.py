import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object (model, transformer, etc.) to a file using pickle.
    This is essential for deploying ML models because you must store the 
    trained model permanently so it can be loaded later without retraining.
    """
    try:
        # Extract the directory path from the provided file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and dump the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Save the object using pickle

    except Exception as e:
        # Wrap any exception in our custom exception class
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    This function trains multiple ML models, performs hyperparameter tuning, 
    and evaluates each model using R² score.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - models: Dictionary of model objects to train
    - param: Dictionary of hyperparameters for each model

    Workflow:
    1. Loop through each model.
    2. Perform GridSearchCV to find best hyperparameters.
    3. Train model with best settings.
    4. Predict on train and test data.
    5. Calculate R² score.
    6. Store test score in report.
    """
    try:
        report = {}  # Final dictionary to store model names and their scores

        for i in range(len(list(models))):
            # Select model by index
            model = list(models.values())[i]

            # Select corresponding hyperparameters
            para = param[list(models.keys())[i]]

            # GridSearchCV searches for the best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Once best params are found, apply them to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train final model with best settings

            # Make predictions on both training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate model performance using R² score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score using model name as key
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raise detailed custom exception if something fails
        raise CustomException(e, sys)
    


def load_object(file_path):
    """
    Loads a previously saved Python object (e.g., ML model) from a file.
    This is crucial for deployment: the application loads a saved model 
    and uses it to make predictions.
    """
    try:
        # Open the file in read-binary mode and load the object
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
