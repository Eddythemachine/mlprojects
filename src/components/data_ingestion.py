import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# ------------------------------------------------------------------------------
# CONFIG CLASS (HOLDS FILE PATHS)
# ------------------------------------------------------------------------------
# This class stores paths where different versions of the dataset (raw, train, test)
# will be saved. Using a dataclass gives convenient default values and removes the
# need for writing an __init__ method manually.
# ------------------------------------------------------------------------------
@dataclass
class DataIngestionConfig:
    # Location where the training dataset will be stored
    train_data_path: str = os.path.join('artifacts', "train.csv")

    # Location where the testing dataset will be stored
    test_data_path: str = os.path.join('artifacts', "test.csv")

    # Location where the unmodified raw dataset will be stored
    raw_data_path: str = os.path.join('artifacts', "data.csv")



# ------------------------------------------------------------------------------
# MAIN DATA INGESTION CLASS
# ------------------------------------------------------------------------------
# This class is responsible for:
# 1. Reading raw data from a CSV
# 2. Creating the artifacts directory
# 3. Saving the raw dataset
# 4. Splitting the dataset into train and test sets
# 5. Saving train.csv and test.csv for the next pipeline steps
# ------------------------------------------------------------------------------
class DataIngestion:
    def __init__(self):
        # Create configuration object so the class can access predefined file paths
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            # ----------------------------------------------------------------------
            # STEP 1: READ THE RAW DATASET
            # ----------------------------------------------------------------------
            # We load the dataset from a CSV file. This is the first step of the pipeline.
            # Note: r"..." ensures the path works properly on Windows.
            # ----------------------------------------------------------------------
            df = pd.read_csv(r"src/notebook/data/stud.csv")
            logging.info('Read the dataset as dataframe')



            # ----------------------------------------------------------------------
            # STEP 2: ENSURE ARTIFACT FOLDER EXISTS
            # ----------------------------------------------------------------------
            # The artifacts folder acts like a "pipeline output" directory
            # where all generated files will be saved for later stages.
            # ------------------------------------------------------------------------------
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)



            # ----------------------------------------------------------------------
            # STEP 3: SAVE THE RAW DATA
            # ----------------------------------------------------------------------
            # This is extremely useful because:
            # - You keep a permanent copy of the original dataset
            # - Future debugging becomes easier
            # - Reproducibility is guaranteed
            # ----------------------------------------------------------------------
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)



            # ----------------------------------------------------------------------
            # STEP 4: TRAIN-TEST SPLIT
            # ------------------------------------------------------------------------------
            # Splitting is important to evaluate the model's generalization ability.
            # We set random_state=42 to ensure the split is reproducible every time
            # the pipeline is run.
            # ------------------------------------------------------------------------------
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)



            # ----------------------------------------------------------------------
            # STEP 5: SAVE THE TRAINING DATASET
            # ------------------------------------------------------------------------------
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)



            # ----------------------------------------------------------------------
            # STEP 6: SAVE THE TESTING DATASET
            # ------------------------------------------------------------------------------
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)



            # ----------------------------------------------------------------------
            # STEP 7: LOGGING COMPLETION
            # ------------------------------------------------------------------------------
            logging.info("Ingestion of the data is completed")



            # ----------------------------------------------------------------------
            # STEP 8: RETURN PATHS FOR NEXT PIPELINE STAGE
            # ------------------------------------------------------------------------------
            # We return train and test file paths so the next components
            # (data transformation and model training) can easily load them.
            # ------------------------------------------------------------------------------
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Wrap any error with a custom exception class for cleaner error tracing.
            raise CustomException(e, sys)



# ------------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ------------------------------------------------------------------------------
# When we run this script directly:
# 1. Data is ingested
# 2. Data is transformed (scaling, encoding, preprocessing)
# 3. A model is trained on the transformed data
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # (1) DATA INGESTION
    # -----------------------------
    # This will create train.csv, test.csv, and data.csv in the artifacts folder.
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()


    # -----------------------------
    # (2) DATA TRANSFORMATION
    # -----------------------------
    # This will load the train/test data, apply preprocessing steps (scaling/encoding),
    # and return numpy arrays ready for model training.
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)


    # -----------------------------
    # (3) MODEL TRAINING
    # -----------------------------
    # This stage fits multiple ML models, compares performance, and returns the best model score.
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
