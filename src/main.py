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

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    # GET BASELINE PREFORMANCE LATER
    # preprocessor_baseline = DataPreProcessor(raw_data)

    preprocessor_univariate = DataPreProcessor(raw_data)
    SSMSP = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate)
    MSMSP = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate)

    preprocessor_RSRQ = DataPreProcessor(raw_data, include_features=["RSRQ"])
    SSMSP_RSRQ = SingleSelectionMultistagePredictor(preprocessor=preprocessor_RSRQ, model_name="SSMSP_RSRQ")
    MSMSP_RSRQ = MultiSelectionMultistagePredictor(preprocessor=preprocessor_RSRQ, model_name="MSMSP_RSRQ")

    preprocessor_RSRP = DataPreProcessor(raw_data, include_features=["RSRP"])
    SSMSP_RSRP = SingleSelectionMultistagePredictor(preprocessor=preprocessor_RSRP, model_name="SSMSP_RSRP")
    MSMSP_RSRP = MultiSelectionMultistagePredictor(preprocessor=preprocessor_RSRP, model_name="MSMSP_RSRP")

    preprocessor_SNR = DataPreProcessor(raw_data, include_features=["SNR"])
    SSMSP_SNR = SingleSelectionMultistagePredictor(preprocessor=preprocessor_SNR, model_name="SSMSP_SNR")
    MSMSP_SNR = MultiSelectionMultistagePredictor(preprocessor=preprocessor_SNR, model_name="MSMSP_SNR")

    preprocessor_CQI = DataPreProcessor(raw_data, include_features=["CQI"])
    SSMSP_CQI = SingleSelectionMultistagePredictor(preprocessor=preprocessor_CQI, model_name="SSMSP_CQI")
    MSMSP_CQI = MultiSelectionMultistagePredictor(preprocessor=preprocessor_CQI, model_name="MSMSP_CQI")
    
    # Increasing history length to 15
    preprocessor_univariate_h15h5 = DataPreProcessor(raw_data, history=15, horizon=5)
    SSMSP_h15h5 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h15h5, model_name="SSMSP_h15h5")
    MSMSP_h15h5 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h15h5, model_name="MSMSP_h15h5")

    # Increasing history length to 20
    preprocessor_univariate_h20h5 = DataPreProcessor(raw_data, history=20, horizon=5)
    SSMSP_h20h5 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h20h5, model_name="SSMSP_h20h5")
    MSMSP_h20h5 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h20h5, model_name="MSMSP_h20h5")

    # Increasing horizon length to 10
    preprocessor_univariate_h10h10 = DataPreProcessor(raw_data, history=10, horizon=10)
    SSMSP_h10h15 = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h10h10, model_name="SSMSP_h10h10")
    MSMSP_h10h15 = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate_h10h10, model_name="MSMSP_h10h10")

    models_to_test = [SSMSP, MSMSP, SSMSP_RSRQ, MSMSP_RSRQ, SSMSP_RSRP, MSMSP_RSRP, SSMSP_SNR, MSMSP_SNR, SSMSP_CQI, MSMSP_CQI, SSMSP_h15h5, MSMSP_h15h5, SSMSP_h20h5, MSMSP_h20h5, SSMSP_h10h15, MSMSP_h10h15]

    with open("Datasets/new_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "test_size"])
        for model in models_to_test:
            model.pre_process()
            model.build_and_train()
            model.test()
            results = model.get_performance_metrics()
            writer.writerow(results)