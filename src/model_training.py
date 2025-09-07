import pandas as pd
import os
import numpy as np
import pickle
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import yaml


log_dir = "logs"
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path : str) -> pd.DataFrame:
    """ 
    Load data  from csv file
    :params file path : path to csv file
    :return: LOaded daatframe

    """

    try:
        df = pd.read_csv(file_path)
        logger.debug("data loaded from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse csv file %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("file not found %s",e)
        raise
    except Exception as e:
        logger.error("unexcepted error occured  while loading the data %s",e)
        raise

def train_model(X_train : np.ndarray,y_train : np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train tghe RandomForest Model
    :params X_train: Training Feature
    :param Y_train: Training Label
    :param params:Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number o fthe sample x_train and y_train must be same")
        
        logger.debug("initilizig XGBClassifier model  with parameters %s",params)
        xg_model = XGBClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_lambda=params['reg_lambda'],
        reg_alpha=params['reg_alpha'],
        objective=params['objective'],
        eval_metric=params['eval_metric'],
        n_jobs=params['n_jobs'],
        random_state=params['random_state'])

        # clf = RandomForestClassifier(n_estimators=params["n_estimators"],random_state=params["random_state"])

        logger.debug("Model training started with %d samples",X_train.shape[0])
        xg_model.fit(X_train,y_train)
        logger.debug("Model training completed")

        return xg_model
    except ValueError as e:
        logger.error("Value error during training %s",e)
        raise
    except Exception as e:
        logger.error("error during model training %s",e)
        raise

def save_model(model,file_path:str) -> None:
    """
    Save the trained model to a file
    :param model : Trained model object
    :param file path: path to save the model file 
    
    """
    try :
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open (file_path,"wb") as file:
            # joblib.dump(model, file)
            pickle.dump(model,file)
        logger.debug("Model saved to %s", file_path)

    except FileNotFoundError as e:
        logger.error("file path not found %s",e)
        raise
    except Exception as e:
        logger.error("error occured while saving the model %s",e)
        raise

def main():

    try:
        params = load_params('params.yaml')['model_training']
        train_data = pd.read_csv("./data/interim/train_processed_data")
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        xg_model = train_model(X_train=X_train,y_train=y_train,params=params)

        model_save_path = "models/model.pkl"
        save_model(xg_model,model_save_path)

    except Exception as e:
        logger.error("Failed  to complete the model building process %s",e)
        print(f"Error : {e}")

if __name__ == "__main__":
    main()