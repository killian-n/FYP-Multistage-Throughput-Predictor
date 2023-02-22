import configparser
import sys
import pandas as pd
import csv
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)


from data_transformation.preprocessor import DataPreProcessor
from models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from models.multi_selection_multistage_predictor import MultiSelectionMultistagePredictor
from models.simple_LSTM import SimpleLSTM
from models.complex_LSTM import ComplexLSTM

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")

    # Univariate
    preprocessor_univariate = DataPreProcessor(raw_data)
    multi_one_univariate = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate)
    multi_all_univariate = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate)
    base_univariate = ComplexLSTM(preprocessor=preprocessor_univariate, model_name="base_univariate")

    # 1
    # Multivariate with RSRQ
    preprocessor_multi1 = DataPreProcessor(raw_data, include_features=["RSRQ"], scaler_file_name="multi_1_scaler.sav")
    multi_one_multi1 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi1, model_name="multi_one_1")
    multi_all_multi1 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi1, model_name="multi_all_1")
    base_multi1 = ComplexLSTM(preprocessor=preprocessor_multi1, model_name="base_1")

    # 2
    # Multivariate with RSRQ, Longitude, Latitude, State, State
    preprocessor_multi2 = DataPreProcessor(raw_data, include_features=["RSRQ", "Longitude", "Latitude", "State"], scaler_file_name="multi_2_scaler.sav")
    multi_one_multi2 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi2, model_name="multi_one_2")
    multi_all_multi2 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi2, model_name="multi_all_2")
    base_multi2 = ComplexLSTM(preprocessor=preprocessor_multi2, model_name="base_2")

    # 3
    # Multivariate with RSRQ, Longitude, Latitude, State, State, NRxRSRQ
    preprocessor_multi3 = DataPreProcessor(raw_data, include_features=["RSRQ", "Longitude", "Latitude", "State", "NRxRSRQ"], scaler_file_name="multi_3_scaler.sav")
    multi_one_multi3 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_multi3, model_name="multi_one_3")
    multi_all_multi3 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_multi3, model_name="multi_all_3")
    base_multi3 = ComplexLSTM(preprocessor=preprocessor_multi3, model_name="base_3")

    models_to_test = [base_univariate, multi_one_univariate, base_multi1, multi_one_multi1, base_multi2, multi_one_multi2, base_multi3, multi_one_multi3, multi_all_univariate, multi_all_multi1, multi_all_multi2, multi_all_multi3]

    # models_to_test = [base_multi1]
    with open(config["metrics"]["RESULTS_PATH"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "mape", "model_size", "accuracy"])
    for model in models_to_test:
        model.pre_process()
        try:
            model.build_and_train()
        except:
            model.build_model()
            model.train()
        model.test()