import sys
import pandas as pd
import configparser
import argparse

config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from data_transformation.preprocessor import DataPreProcessor

if __name__ == "__main__":
    raw_data = pd.read_csv(config["global"]["PROJECT_PATH"]+"/Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    parser = argparse.ArgumentParser(description='Create Train test datasets')
    parser.add_argument('--prefix', type=str,
                         help="The file prefix of the saved train and test datasets found in Datasets directory. \n Assumed to be same as prefix if not provided")
    parser.add_argument('--include',nargs="+", type=str,
                         help="List of features to included for datasets.")
    parser.add_argument('--history_window', type=int, help="Length of the history window in seconds")
    parser.add_argument('--horizon_window', type=int, help="Length of the horizon window in seconds")
    parser.add_argument('--env_path', type=str,
                         help="List of features to included for datasets.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    prefix = args.prefix
    history = args.history_window
    include_features = args.include
    horizon = args.horizon_window
    if not prefix:
        print("please include a prefix for the name of the saved datasets, scaler and imputer.")
        sys.exit(1)
    
    if not history:
        history = 10
        
    if not horizon:
        horizon = 5

    preprocesor = DataPreProcessor(dataframe=raw_data, name=prefix, include_features=include_features, history=history, horizon=horizon)
