import shutil
import pandas as pd
from data_cleaning import clean_data
from feature_construction import construct_features
from feature_selction import selction
from train import training

def main():
    source_path = 'data/train-test.csv'
    copy_path = 'data/train-test-copy.csv'
    
    print(f"Making a copy of {source_path} to {copy_path}...")
    shutil.copy(source_path, copy_path)
    
    print("Loading copied dataset for inplace feature engineering...")
    df = pd.read_csv(copy_path)

    print("<------Data cleaning started------>")
    try:
        clean_data(df)
    except Exception as e:
        print(f"Error {e} while performing data cleaning")
    print("<------Data cleaning completed-------->\n")
    
    print("<------Features construction started------>")
    try:
        construct_features(df)
    except Exception as e:
        print(f"Error {e} while performing feature constructio")
    print("<------Features construction completed------>")
    
    print("<------feature selction started------->\n")
    try:
        df = selction(df)
        print(f"Saving processed dataframe inplace back to {copy_path}...")
        df.to_csv(copy_path, index=False)
    except Exception as e:
        print(f"Error {e} while performing feature selection")
    print("<------feature selction completed------->\n")

    print("<------Training started------>")
    try:    
        training(df)
    except Exception as e:
        print(f"Error {e} while training and testing")
    print("<------Training completed------>")
    

if __name__ == '__main__':
    main()
