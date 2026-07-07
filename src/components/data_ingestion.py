#this file aims to load data from diffrent source

#imports
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import custom_exception
from sklearn.model_selection import train_test_split

@dataclass #decorator used to say this dataclass only saves metadata
class data_ingestion_metadata:
    train_dataset_path:str = os.path.join('artifacts','train_dataset.csv') #path where data will be stored
    test_dataset_path:str = os.path.join('artifacts','test_dataset.csv')
    raw_dataset_path:str = os.path.join('artifacts','raw_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.Ingestion_obj = data_ingestion_metadata()

    #data loading and ingestion pipeline
    def DataIngestion_pipeline(self):
        logging.info("DataIngestion Process has been initialized")
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            path = os.path.join(base_dir, 'Dataset', 'indian_railway_failure_detection_maintenance_v2.csv')
            df = pd.read_csv(path)
            logging.info('data loaded sucessfully')

            #to create a artifact folder from self obj path
            os.makedirs(os.path.dirname(os.path.join(self.Ingestion_obj.train_dataset_path)),exist_ok=True)
            df.to_csv(self.Ingestion_obj.raw_dataset_path,index=False,header=True)

            train,test = train_test_split(df,test_size=0.3,random_state=42)

            train.to_csv(self.Ingestion_obj.train_dataset_path,index=False,header=True)
            test.to_csv(self.Ingestion_obj.test_dataset_path,index=False,header=True)
            logging.info('data ingestion completed for train and test data"')

            return {
                self.Ingestion_obj.test_dataset_path,
                self.Ingestion_obj.train_dataset_path
            }
        
        except Exception as e:
            raise custom_exception(e,sys)
