import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix, precision_score, recall_score,f1_score,roc_auc_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)