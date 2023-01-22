import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor
from models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from models.multi_selection_multistage_predictor import MultiSelectionMultistagePredictor

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")

    preprocessor_univariate = DataPreProcessor(raw_data) 
    SSMSP = SingleSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_univariate)
    MSMSP = MultiSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_univariate)

    preprocessor_cqi = DataPreProcessor(raw_data, include_features=["CQI"])
    SSMSP_CQI = SingleSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_cqi, model_name="SSMSP_CQI")
    MSMSP_CQI = MultiSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_cqi, model_name="MSMSP_CQI")

    preprocessor_snr = DataPreProcessor(raw_data, include_features=["SNR"])
    SSMSP_SNR = SingleSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_snr, model_name="SSMSP_SNR")
    MSMSP_SNR = MultiSelectionMultistagePredictor(raw_data, preprocessor=preprocessor_snr, model_name="MSMSP_SNR")

    models_to_test = [SSMSP, MSMSP, SSMSP_CQI, MSMSP_CQI, SSMSP_SNR, MSMSP_SNR]

    with open("Datasets/results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "test_size"])
        for model in models_to_test:
            model.build_and_train()
            model.test()
            results = model.get_performance_metrics()
            writer.writerow(results)          



