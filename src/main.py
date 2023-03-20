import configparser
import sys
import pandas as pd
import csv
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)


from data_transformation.preprocessor import DataPreProcessor
from training_models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from training_models.multi_selection_multistage_predictor import MultiSelectionMultistagePredictor
from training_models.baseline_regression_model import BaselineLSTM
from training_models.optimized_models import optimizedBaseline

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")

    # Univariate
    preprocessor_univariate = DataPreProcessor(raw_data)
    multi_one_univariate = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate, model_name="univariate_multiOne")
    univariate_baseline = BaselineLSTM(preprocessor=preprocessor_univariate, model_name="univariate_baseline")
    models_to_test = [multi_one_univariate, univariate_baseline]

    # All network features
    # all_network_preprocessor = DataPreProcessor(raw_data, scaler_file_name="all_network_features.sav", include_features=["NRxRSRQ", "RSRQ","RSRP" ,"SNR", "CQI", "RSSI", "NRxRSRP"])
    # all_network_baseline = optimizedBaseline(preprocessor=all_network_preprocessor, model_name="all_network_baseline")
    # all_network_multiOne = SingleSelectionMultistagePredictor(preprocessor=all_network_preprocessor, model_name="all_network_multiOne")
    # models_to_test = [all_network_baseline]

    # with open(config["metrics"]["RESULTS_PATH"], "a", newline="") as f:
    #     writer = csv.writer(f)
       #  writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "mape", "model_size", "accuracy"])
    for model in models_to_test:
        model.pre_process()
        try:
            model.build_and_train()
        except:
            model.build_model()
            model.train()
        model.test()

    # all_network_multiAll = MultiSelectionMultistagePredictor(preprocessor=all_network_preprocessor,
    #                                                         pretrained_sequence_models="all_network_multiOne",
    #                                                             model_name="all_network_multiAll")
    # all_network_multiAll.pre_process()
    # all_network_multiAll.build_and_train()
    # all_network_multiAll.test()
    univariate_multiAll = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate,
                                                             model_name="univariate_multiAll", pretrained_sequence_models="univariate_multiOne")
    univariate_multiAll.pre_process()
    univariate_multiAll.build_and_train()
    univariate_multiAll.test()