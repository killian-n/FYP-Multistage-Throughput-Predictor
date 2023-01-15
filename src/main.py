import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor
from models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from models.multi_selection_multistage_predictor import MultiSelectionMultistagePredictor

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    preprocessor = DataPreProcessor(raw_data) 
    SSMSP = SingleSelectionMultistagePredictor(raw_data, preprocessor=preprocessor)
    MSMSP = MultiSelectionMultistagePredictor(raw_data, preprocessor=preprocessor)