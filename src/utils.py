#used for utility function

import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl
from src.logger import logging
from src.exception import custom_exception
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#save model 
def save_objects(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pkl.dump(obj,file_obj)

        logging.info(f'pickle file has been saved at {file_path}')
    
    except Exception as e:
        raise custom_exception(e,sys)
    

#to evaluate the model
def eval_model(x_train, x_test, y_train, y_test, models):
    
    report = {}
    
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        score = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report[model_name] = score
    
    return report


#to load file
def load_object(file_path):
    try:
        with open (file_path, 'rb') as file:
            return pkl.load(file)
        
    except Exception as e:
        raise custom_exception(e,sys)
