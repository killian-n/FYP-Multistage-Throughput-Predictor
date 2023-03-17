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
    multi_one_univariate = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate, model_name="unop_univariate_multiOne")
    univariate_baseline = BaselineLSTM(preprocessor=preprocessor_univariate, model_name="unop_univariate_baseline")

    # # 1
    # # Multivariate with RSRQ
    # preprocessor_multi1 = DataPreProcessor(raw_data, include_features=["RSRQ"], scaler_file_name="multi_1_scaler.sav")
    # multi_one_multi1 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi1, model_name="multi_one_1")
    # base_multi1 = ComplexLSTM(preprocessor=preprocessor_multi1, model_name="base_1")

    # 2
    # Multivariate with RSRQ, Longitude, Latitude, State, State
    # preprocessor_multi2 = DataPreProcessor(raw_data, include_features=["RSRQ", "Longitude", "Latitude", "State"], scaler_file_name="multi_2_scaler.sav")
    # multi_one_multi2 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi2, model_name="multi_one_2")
    # base_multi2 = ComplexLSTM(preprocessor=preprocessor_multi2, model_name="base_2")

    # 3
    # Multivariate with RSRQ, Longitude, Latitude, State, NRxRSRQ
    # preprocessor_multi3 = DataPreProcessor(raw_data, include_features=["RSRQ", "Longitude", "Latitude", "State", "NRxRSRQ"], scaler_file_name="multi_3_scaler.sav")
    # multi_one_multi3 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi3, model_name="multi_one_3")
    # base_multi3 = ComplexLSTM(preprocessor=preprocessor_multi3, model_name="base_3")

    # multi_all_univariate = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate,model_name="multi_all_univariate", pretrained_sequence_models="multi_one_univariate")
    # multi_all_multi1 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi1, model_name="multi_all_1", pretrained_sequence_models="multi_one_1")
    # multi_all_multi2 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi2, model_name="multi_all_2", pretrained_sequence_models="multi_one_2")
    # multi_all_multi3 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi3, model_name="multi_all_3", pretrained_sequence_models="multi_one_3")
    
    # models_to_test = [base_univariate, multi_one_univariate, base_multi1, multi_one_multi1, base_multi2, multi_one_multi2, base_multi3,
    #  multi_one_multi3, multi_all_univariate, multi_all_multi1, multi_all_multi2, multi_all_multi3]

    # 4 
    # preprocessor_multi4 = DataPreProcessor(raw_data, include_features=["RSRQ", "NRxRSRQ", "SNR", "UL_bitrate", "NetworkMode", "State"], scaler_file_name="multi_4_scaler.sav")
    # multi_one_multi4 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi4, model_name="multi_one_4")
    # base_multi4 = ComplexLSTM(preprocessor=preprocessor_multi4, model_name="base_4")
    # multi_all_multi4 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi4, model_name="multi_all_4", pretrained_sequence_models="multi_one_4")


    # 5
    all_network_preprocessor = DataPreProcessor(raw_data, scaler_file_name="all_network_features.sav", include_features=["NRxRSRQ", "RSRQ","RSRP" ,"SNR", "CQI", "RSSI", "NRxRSRP"])
    # all_network_baseline = BaselineLSTM(preprocessor=all_network_preprocessor, model_name="unop_all_network_baseline")
    all_network_multiOne = SingleSelectionMultistagePredictor(preprocessor=all_network_preprocessor, model_name="all_network_multiOne")

    models_to_test = [all_network_multiOne]

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

    all_network_multiAll = MultiSelectionMultistagePredictor(preprocessor=all_network_preprocessor,
                                                            pretrained_sequence_models="all_network_multiOne",
                                                                model_name="all_network_multiAll")
    all_network_multiAll.pre_process()
    all_network_multiAll.build_and_train()
    all_network_multiAll.test()
    # univariate_multiAll = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate,
    #                                                          model_name="unop_univariate_multiAll", pretrained_sequence_models="unop_univariate_multiOne")
    # univariate_multiAll.pre_process()
    # univariate_multiAll.build_and_train()
    # univariate_multiAll.test()