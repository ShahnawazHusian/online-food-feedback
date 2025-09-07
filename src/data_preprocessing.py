import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer

log_dir = "logs"
os.makedirs(log_dir,exist_ok = True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocessing the DataFrame by encoding the traget columns  """
    try:
        logger.debug("starting preprocessing for DataFrame")
        le = LabelEncoder()
        df["Feedback"] = le.fit_transform(df["Feedback"])

        ohe = OneHotEncoder(sparse_output=False,drop="first")
        oe = OrdinalEncoder()

        transformer = ColumnTransformer(transformers=[
            ("trf1", ohe, ["Gender", "Marital Status", "Occupation"]),
            ("trf2", OrdinalEncoder(categories=[
                ["No Income", "Below Rs.10000", "10001 to 25000", "25001 to 50000", "More than 50000"],
                ["Uneducated", "School", "Graduate", "Post Graduate", "Ph.D"]
            ]), ["Monthly Income", "Educational Qualifications"])
        ], remainder="passthrough")

        df_encoded = transformer.fit_transform(df)
        df = pd.DataFrame(df_encoded)
       
        
        logger.debug("Target columns encoded")
        return df
        
    except KeyError as e:
        logger.error("column not found %s",e)
        raise    
    except Exception as e:
        logger.error("Error during encoding %s", e)
        raise

def main():
    """ Main function to load raw data , process it and save processed data """
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")
        
        train_processed_data = preprocess_df(train_data)
        test_processed_data = preprocess_df(test_data)

        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed_data"),index = False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed_data"),index = False)

        logger.debug("processed data saved %s",data_path)

    except FileNotFoundError as e:
        logger.error("file not found %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No data %s",e)
        raise
    except Exception as e:
        logger.error("failed to complete the data transformation process : %s",e)
        print(f"Error : {e}")

if __name__ == "__main__":
    main()