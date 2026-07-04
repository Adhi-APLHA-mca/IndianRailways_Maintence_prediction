#this file aims to load data from diffrent source

#imports
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import logging


@dataclass #decorator used to say this dataclass only saves metadata
class data_ingestion_metadata:
    train_dataset_path:str = os.path.join('artifacts','train_dataset.csv') #path where data will be stored
    test_dataset_path:str = os.path.join('artifacts','test_dataset.csv')
    raw_dataset_path:str = os.path.join('artifacts','raw_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.obj = data_ingestion_metadata()

    #data loading and ingestion pipeline
    def DataIngestion_pipeline(self):
        logging.info("DataIngestion Process has been initialized")
        try:
            df = pd.read_csv(input("please place the path of the csv: "))
            
        except:
            pass

if __name__ == "__main__":

    obj = DataIngestion()
    obj.DataIngestion_pipeline()