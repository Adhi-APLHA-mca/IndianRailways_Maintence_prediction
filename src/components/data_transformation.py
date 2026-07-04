#this file aims to have preprocess the data for the model training and testing

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from src.logger import logging
from src.exception import custom_exception
from src.components.data_ingestion import DataIngestion
from src.utils import save_objects



#this data class has been used to save pickle file of preprocessing engine
@dataclass
class DataTransformationConfig:
    data_preprocessing_file:str = os.path.join('artifacts','data_preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.processing_obj = DataTransformationConfig()


    #setup function for category and numerical pipeline to perform standard scaler/imputation and ohe hot encoding
    def DataTransformation_settings_pipeline(self, train_df):
        try:
            #spliting features
            cat_features = train_df.select_dtypes(include="object").columns
            num_features = train_df.select_dtypes(exclude="object").columns

            #creating a pipeline for preprocessing
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',KNNImputer(n_neighbors = 2)),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ('ohe',OneHotEncoder(sparse_output=False))
                ]
            )

            preprocessing = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )
            
            logging.info('Data transformation pipeline settings has been completed')

            return preprocessing
        

        except Exception as e:
            raise custom_exception(e,sys)
        
    
    #actuall data transformation pipeline
    def DataTransformation_pipeline(self,train_path,test_path):
        try:
            DataTransformation_train_df = pd.read_csv(train_path)
            DataTransformation_test_df = pd.read_csv(test_path)

            logging.info('Data transformation pipeline has been initiated')

            target_feature = 'maintenance_required'

            x_train = DataTransformation_train_df.drop(columns=[target_feature])
            y_train = DataTransformation_train_df[target_feature]

            x_test = DataTransformation_test_df.drop(columns=[target_feature])
            y_test = DataTransformation_test_df[target_feature]

            #applying preprocessing pipeline to the train and test data
            preprocessor = self.DataTransformation_settings_pipeline(x_train)

            x_train = preprocessor.fit_transform(x_train)
            x_test = preprocessor.transform(x_test)

            save_objects(
                file_path=self.processing_obj.data_preprocessing_file,
                obj=preprocessor
            )

            logging.info('Data transformation pipeline objects has been stored in form of pkl')

            return(
                x_train,
                x_test,
                y_train,
                y_test,
                preprocessor
            )
        
        except Exception as e:
            raise custom_exception(e,sys)

