#this file aims to load data from diffrent source

#imports
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import custom_exception

class DataLoading:

    #data loading and ingestion pipeline
    def DataIngestion_pipeline(self):
        logging.info("DataIngestion Process has been initialized")
        try:
            df = pd.read_csv(input("please place the path of the csv: "))
            
            cat_features = df.select_dtypes(include='object')
            num_features = df.select_dtypes(exclude='object')

            return df,cat_features,num_features
        except Exception as e:
            raise custom_exception (e,sys)
