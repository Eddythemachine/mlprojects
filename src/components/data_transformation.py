import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Location where the preprocessor (the full transformation pipeline) will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


class DataTransformation:
    """
    This class handles **all data preprocessing steps**:
    - handling missing values
    - encoding categorical features
    - scaling numeric features
    - combining all transformations into one pipeline
    - saving the pipeline for future use (deployment)
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds the preprocessing pipeline.
        The pipeline has two branches:
          1. Numerical pipeline
          2. Categorical pipeline
        Both pipelines are combined using ColumnTransformer.

        This prepares your data consistently every time.
        """
        try:
            # Numerical columns to scale
            numerical_columns = ["writing_score", "reading_score"]

            # Categorical columns that require encoding
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numeric columns:
            # 1. Replace missing values with median
            # 2. Scale values using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical columns:
            # 1. Replace missing values with the most frequent value
            # 2. Encode categories using OneHotEncoder
            # 3. Scale encoded values (without_mean=True because sparse matrices cannot center)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # ColumnTransformer applies:
            # - num_pipeline to numerical columns
            # - cat_pipeline to categorical columns
            # Result: one combined processed dataset
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        """
        Loads training & testing data,
        Applies preprocessing pipeline,
        Saves the preprocessor for reuse,
        Returns transformed arrays ready for model training.
        """
        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Build the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Target/output variable
            target_column_name = "math_score"

            # Numeric columns (used again)
            numerical_columns = ["writing_score", "reading_score"]

            # Separate features (X) from target (y)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and test data.")

            # Fit on training data and transform both sets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine processed features with target value into one array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            # Save the preprocessor for future use (deployment)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return processed arrays + path to saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
