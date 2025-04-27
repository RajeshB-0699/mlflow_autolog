import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import json
import os

import dagshub
dagshub.init(repo_owner='RajeshB-0699', repo_name='mlflow_autolog', mlflow=True)

mlflow.set_experiment('mlflow-autoLogging')
mlflow.set_tracking_uri('https://dagshub.com/RajeshB-0699/mlflow_autolog.mlflow')

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in loading data from filepath : {e}")

def load_params_collection(filepath : str) -> float:
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
            return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"Error in loading data from params file path : {e}")

def load_params_model(filepath : str) -> int:
    try:
        with open(filepath, 'r') as file:
            params = yaml.safe_load(file)
            return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception(f"Error in loading params from params file path : {e}")
    
def split_data(data: pd.DataFrame, test_size : float) -> tuple[pd.DataFrame, pd.Series]:
    try:
        return train_test_split(data, test_size = test_size,  random_state=42)
    except ValueError as e:
        raise ValueError(f"Error in splitting dataset : {e}")

def save_data(data: pd.DataFrame, filepath: str) -> None:
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error in saving data to {filepath} :  {e}")

def fill_missing_with_median(data):
    for column in data.columns:
        if data[column].isnull().any():
            median_value = data[column].median()
            data[column].fillna(median_value, inplace=True)
    return data

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns = ['Potability'], axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error in preparing data : {e}")
    
def train_model(X: pd.DataFrame, y: pd.Series, n_estimators : int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators = n_estimators)
        clf.fit(X,y)
        return clf
    except Exception as e:
        raise Exception(f"Error in training model : {e}")

def save_model(model : RandomForestClassifier, filepath: str) -> None:
    try:
        with open(filepath, "wb")  as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error in saving file to {filepath} : {e}")

def load_model(filepath : str):
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
            return model
    except Exception as e:
        raise Exception(f"Error in loading model from {filepath} : {e}")

def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_pred,y_test)
        precision = precision_score(y_pred, y_test)
        f1  = f1_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "f1": f1,
            "recall" : recall
        }

        return metrics_dict
    
    except Exception as e:
        raise Exception(f"Error in evaluating model : {e}")

def save_metrics(metrics_dict, filepath: str) -> None:
    try:
        with open(filepath,'w') as file:
            json.dump(metrics_dict, file, indent = 4)
    except Exception as e:
        raise Exception(f"Error in saving json file in {filepath} : {e}")
    

def main():
    try:
        datafilepath = "https://raw.githubusercontent.com/DataThinkers/Datasets/refs/heads/main/DS/water_potability.csv"
        params_filepath = "params.yaml"
        train_datapath = "./data/raw/train.csv"
        test_datapath = "./data/raw/test.csv"
        train_processed_datapath = "./data/processed/train_processed.csv"
        test_processed_datapath = "./data/processed/test_processed.csv"
        model_name = "model.pkl"
        metrics_filepath = "metrics.json"

        datapath = os.path.join("data","raw")
        os.makedirs(datapath)

        
        
        processedpath = os.path.join("data","processed")
        os.makedirs(processedpath)

        data = load_data(datafilepath)
        test_size = load_params_collection(params_filepath)
        n_estimators = load_params_model(params_filepath)
        train_data, test_data = split_data(data, test_size = test_size)
        save_data(train_data, os.path.join(datapath, "train.csv"))
        save_data(test_data, os.path.join(datapath,"test.csv"))

        train_data = load_data(train_datapath)
        test_data = load_data(test_datapath)

        train_processed = fill_missing_with_median(train_data)
        test_processed = fill_missing_with_median(test_data)

        save_data(train_processed, os.path.join(processedpath,"train_processed.csv"))
        save_data(test_processed, os.path.join(processedpath,"test_processed.csv"))

        train_processed_data = load_data(train_processed_datapath)
        test_processed_data = load_data(test_processed_datapath)
        X_train, y_train = prepare_data(train_processed_data)
        model = train_model(X_train,y_train, n_estimators = n_estimators)
        save_model(model, model_name)
        X_test , y_test = prepare_data(test_processed_data)
        model_loaded = load_model(model_name)
        evaluation_dict = evaluation_model(model_loaded, X_test, y_test)
        save_metrics(evaluation_dict, metrics_filepath)

    except Exception as e:
        raise Exception(f"Error in processing all: {e}")


if __name__ == "__main__":
    main()