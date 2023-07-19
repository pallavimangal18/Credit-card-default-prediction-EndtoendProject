# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score,f1_score,roc_auc_score   
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )    
            models = {
            'LogisticRegression' : LogisticRegression(),
            'DecisionTreeClassifier' : DecisionTreeClassifier(),
            'RandomForestClassifier' : RandomForestClassifier(),
            'SVC' : SVC(),
            'XGBClassifier' : XGBClassifier()

             }
            
            logging.info("Defining hyper parameter tuning")

            params = {
                "LogisticRegression" : {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },

                'DecisionTreeClassifier' : {
                    'criterion' : ['gini','entropy','log_loss'],
                    'splitter' : ['best' , 'random']

                },

                "RandomForestClassifier" : {
                    'n_estimators':[75,100,125],
                    'min_samples_leaf':[20,25,30],
                    'max_depth':[5,7]},

                'SVC' : {
                    'C' : [0.1,1],
                    'kernel' : ['linear' , 'rbf'],
                    #'gamma' : ['scale' , 'auto']

                },

                "XGBClassifier" :{
                    'learning_rate':[0.05,0.075,0.1],
                    'max_depth':[4,5,6],
                    'colsample_bytree':[0.7,0.8],
                    'n_estimators':[75,100,125],
                    'colsample_bylevel':[0.7,0.8]
                }


            }
            
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')


            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy_score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy_Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )


            predicted=best_model.predict(X_test)

            Accuracy_score = accuracy_score(y_test, predicted)
            return Accuracy_score




        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)    