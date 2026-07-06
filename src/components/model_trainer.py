import pandas as pd
import numpy as np
import sys 
import os

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from src.utils import eval_model
from src.utils import save_objects
from src.exception import custom_exception
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    ModelTrainerPath:str = os.path.join("artifacts",'model.pkl')

class ModelTrainer():
    def __init__(self):
        self.trainer_obj = ModelTrainerConfig()

    #function to check for models
    def ModelTrainer_Pipeline(self,x_test,x_train,y_test,y_train):
        logging.info("Model training started")
        try:
            models = {
                        "Logistic Regression": LogisticRegression(class_weight='balanced'),
                        "K-Neighbors Classifier": KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto'),
                        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
                        "Random Forest Classifier": RandomForestClassifier(class_weight='balanced'),
                        "XGBClassifier": XGBClassifier(),
                        "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                        "AdaBoost Classifier": AdaBoostClassifier(),
                        "SVC": SVC(class_weight='balanced')
                    }

            model_report:dict= eval_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)

            best_model_score = (max(sorted(model_report.values())))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            print(model_report)

            if best_model_score < 0.6:
                raise custom_exception('No best model found! all the model accuracy score are below 60%')
            
            best_model = models[best_model_name]
            
            logging.info(f"best model found for both training and test data best model - {best_model_name} , accuracy score - {best_model_score}")

            save_objects(
                file_path=self.trainer_obj.ModelTrainerPath,
                obj=best_model
            )

        except Exception as e:
            raise custom_exception(e,sys)
