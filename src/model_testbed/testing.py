# fold 1, 4, 6, 8, 9 NO LOW

import configparser
import sys
import pandas as pd
import csv
import numpy as np
import random
from matplotlib import pyplot as plt
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor
from models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from models.multi_selection_multistage_predictor import MultiSelectionMultistagePredictor

K = 10
random_seeds = random.sample(range(200), K)

raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")

SSMSP_results = {"model_name":"SSMSP", "trainable_params":0, "non_trainable_params":0, "train_time":[], "predict_time":[], "mse":[],"mae":[],"average_bias":[], "test_size":[]}
SSMSP_SNR_results = {"model_name":"SSMSP_SNR", "trainable_params":0, "non_trainable_params":0, "train_time":[], "predict_time":[], "mse":[],"mae":[],"average_bias":[], "test_size":[]}
MSMSP_results = {"model_name":"MSMSP", "trainable_params":0, "non_trainable_params":0, "train_time":[], "predict_time":[], "mse":[],"mae":[],"average_bias":[], "test_size":[]}
MSMSP_SNR_results = {"model_name":"MSMSP_SNR", "trainable_params":0, "non_trainable_params":0, "train_time":[], "predict_time":[], "mse":[],"mae":[],"average_bias":[], "test_size":[]}

for i in random_seeds:

    preprocessor_univariate = DataPreProcessor(raw_data, random_seed=i)
    preprocessor_univariate_sparse = DataPreProcessor(raw_data, sparse=True, random_seed=i)
    SSMSP = SingleSelectionMultistagePredictor(preprocessor=preprocessor_univariate_sparse)
    MSMSP = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate)

    preprocessor_SNR = DataPreProcessor(raw_data, include_features=["SNR"], random_seed=i)
    preprocessor_SNR_sparse = DataPreProcessor(raw_data, include_features=["SNR"], sparse=True, random_seed=i)
    SSMSP_SNR = SingleSelectionMultistagePredictor(preprocessor=preprocessor_SNR_sparse, model_name="SSMSP_SNR")
    MSMSP_SNR = MultiSelectionMultistagePredictor(preprocessor=preprocessor_SNR, model_name="MSMSP_SNR")

    SSMSP.pre_process()
    SSMSP.build_and_train()
    SSMSP.test()
    results = SSMSP.get_performance_metrics()
    SSMSP_results["trainable_params"] = results[1]
    SSMSP_results["non_trainable_params"] = results[2]
    SSMSP_results["train_time"].append(results[3])
    SSMSP_results["predict_time"].append(results[4])
    SSMSP_results["mse"].append(results[5])
    SSMSP_results["mae"].append(results[6])
    SSMSP_results["average_bias"].append(results[7])
    SSMSP_results["test_size"].append(results[8])


    SSMSP_SNR.pre_process()
    SSMSP_SNR.build_and_train()
    SSMSP_SNR.test()
    results = SSMSP_SNR.get_performance_metrics()
    SSMSP_SNR_results["trainable_params"] = results[1]
    SSMSP_SNR_results["non_trainable_params"] = results[2]
    SSMSP_SNR_results["train_time"].append(results[3])
    SSMSP_SNR_results["predict_time"].append(results[4])
    SSMSP_SNR_results["mse"].append(results[5])
    SSMSP_SNR_results["mae"].append(results[6])
    SSMSP_SNR_results["average_bias"].append(results[7])
    SSMSP_SNR_results["test_size"].append(results[8])

    MSMSP.pre_process()
    MSMSP.build_and_train()
    MSMSP.test()
    results = MSMSP.get_performance_metrics()
    MSMSP_results["trainable_params"] = results[1]
    MSMSP_results["non_trainable_params"] = results[2]
    MSMSP_results["train_time"].append(results[3])
    MSMSP_results["predict_time"].append(results[4])
    MSMSP_results["mse"].append(results[5])
    MSMSP_results["mae"].append(results[6])
    MSMSP_results["average_bias"].append(results[7])
    MSMSP_results["test_size"].append(results[8])

    MSMSP_SNR.pre_process()
    MSMSP_SNR.build_and_train()
    MSMSP_SNR.test()
    results = MSMSP_SNR.get_performance_metrics()
    MSMSP_SNR_results["trainable_params"] = results[1]
    MSMSP_SNR_results["non_trainable_params"] = results[2]
    MSMSP_SNR_results["train_time"].append(results[3])
    MSMSP_SNR_results["predict_time"].append(results[4])
    MSMSP_SNR_results["mse"].append(results[5])
    MSMSP_SNR_results["mae"].append(results[6])
    MSMSP_SNR_results["average_bias"].append(results[7])
    MSMSP_SNR_results["test_size"].append(results[8])

# Get average results
SSMSP_mse = sum(SSMSP_results["mse"])/K
SSMSP_mae = sum(SSMSP_results["mae"])/K
SSMSP_bias = sum(SSMSP_results["average_bias"])/K
SSMSP_train_time = sum(SSMSP_results["train_time"])/K
SSMSP_predict_time = sum(SSMSP_results["predict_time"])/K
SSMSP_test_size = sum(SSMSP_results["test_size"])/K

SSMSP_SNR_mse = sum(SSMSP_SNR_results["mse"])/K
SSMSP_SNR_mae = sum(SSMSP_SNR_results["mae"])/K
SSMSP_SNR_bias = sum(SSMSP_SNR_results["average_bias"])/K
SSMSP_SNR_train_time = sum(SSMSP_SNR_results["train_time"])/K
SSMSP_SNR_predict_time = sum(SSMSP_SNR_results["predict_time"])/K
SSMSP_SNR_test_size = sum(SSMSP_SNR_results["test_size"])/K

MSMSP_mse = sum(MSMSP_results["mse"])/K
MSMSP_mae = sum(MSMSP_results["mae"])/K
MSMSP_bias = sum(MSMSP_results["average_bias"])/K
MSMSP_train_time = sum(MSMSP_results["train_time"])/K
MSMSP_predict_time = sum(MSMSP_results["predict_time"])/K
MSMSP_test_size = sum(MSMSP_results["test_size"])/K

MSMSP_SNR_mse = sum(MSMSP_SNR_results["mse"])/K
MSMSP_SNR_mae = sum(MSMSP_SNR_results["mae"])/K
MSMSP_SNR_bias = sum(MSMSP_SNR_results["average_bias"])/K
MSMSP_SNR_train_time = sum(MSMSP_SNR_results["train_time"])/K
MSMSP_SNR_predict_time = sum(MSMSP_SNR_results["predict_time"])/K
MSMSP_SNR_test_size = sum(MSMSP_SNR_results["test_size"])/K

with open("Datasets/Multiple_Split_Results_Balanced.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "test_size"])
    writer.writerow(["SSMSP", SSMSP_results["trainable_params"], SSMSP_results["non_trainable_params"],
     SSMSP_train_time, SSMSP_predict_time, SSMSP_mse, SSMSP_mae, SSMSP_bias, SSMSP_test_size])
    
    writer.writerow(["SSMSP_SNR", SSMSP_SNR_results["trainable_params"], SSMSP_SNR_results["non_trainable_params"],
     SSMSP_SNR_train_time, SSMSP_SNR_predict_time, SSMSP_SNR_mse, SSMSP_SNR_mae, SSMSP_SNR_bias, SSMSP_SNR_test_size])

    writer.writerow(["MSMSP", MSMSP_results["trainable_params"], MSMSP_results["non_trainable_params"],
     MSMSP_train_time, MSMSP_predict_time, MSMSP_mse, MSMSP_mae, MSMSP_bias, MSMSP_test_size])

    writer.writerow(["MSMSP_SNR", MSMSP_SNR_results["trainable_params"], MSMSP_SNR_results["non_trainable_params"],
     MSMSP_SNR_train_time, MSMSP_SNR_predict_time, MSMSP_SNR_mse, MSMSP_SNR_mae, MSMSP_SNR_bias, MSMSP_SNR_test_size])

with open("Datasets/RAW_Multiple_Split_Results_Balanced.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_name", "trainable_params", "non_trainable_params", "train_time", "predict_time", "mse", "mae", "average_bias", "test_size"])
    writer.writerow(["SSMSP", SSMSP_results["trainable_params"], SSMSP_results["non_trainable_params"],
     SSMSP_results["train_time"], SSMSP_results["predict_time"], SSMSP_results["mse"], SSMSP_results["mae"], SSMSP_results["average_bias"], SSMSP_results["test_size"]])
    
    writer.writerow(["SSMSP_SNR", SSMSP_SNR_results["trainable_params"], SSMSP_SNR_results["non_trainable_params"],
     SSMSP_SNR_results["train_time"], SSMSP_SNR_results["predict_time"], SSMSP_SNR_results["mse"],
      SSMSP_SNR_results["mae"], SSMSP_SNR_results["average_bias"], SSMSP_SNR_results["test_size"]])

    writer.writerow(["MSMSP", MSMSP_results["trainable_params"], MSMSP_results["non_trainable_params"],
     MSMSP_results["train_time"], MSMSP_results["predict_time"], MSMSP_results["mse"], MSMSP_results["mae"], MSMSP_results["average_bias"], MSMSP_results["test_size"]])

    writer.writerow(["MSMSP_SNR", MSMSP_SNR_results["trainable_params"], MSMSP_SNR_results["non_trainable_params"],
     MSMSP_SNR_results["train_time"], MSMSP_SNR_results["predict_time"], MSMSP_SNR_results["mse"], MSMSP_SNR_results["mae"], MSMSP_SNR_results["average_bias"], MSMSP_SNR_results["test_size"]])

splits = list(range(1,K+1))

fig, ax = plt.subplots()
ax.plot(splits, SSMSP["mse"], label="SSMSP Univariate")
ax.plot(splits, SSMSP_SNR["mse"], label="SSMSP with SNR")
ax.plot(splits, MSMSP["mse"], label="MSMSP Univariate")
ax.plot(splits, MSMSP_SNR["mse"], label="MSMSP with SNR")

ax.plot(splits, np.repeat(SSMSP_mse, K), label="SSMSP Univariate Avg")
ax.plot(splits, np.repeat(SSMSP_SNR_mse, K), label="SSMSP with SNR Avg")
ax.plot(splits, np.repeat(MSMSP_mse, K), label="MSMSP Univariate Avg")
ax.plot(splits, np.repeat(MSMSP_SNR_mse, K), label="MSMSP with SNR Avg")

ax.set_title("Mean Squared Error over Multiple Train/Test Splits", fontsize=20)
ax.set_xlabel("Split", fontsize=14)
ax.set_ylabel("MSE", fontsize=14)
fig.savefig("Images/Univariate_and_SNR_MSE_Balanced.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(splits, SSMSP["mae"], label="SSMSP Univariate")
ax.plot(splits, SSMSP_SNR["mae"], label="SSMSP with SNR")
ax.plot(splits, MSMSP["mae"], label="MSMSP Univariate")
ax.plot(splits, MSMSP_SNR["mae"], label="MSMSP with SNR")

ax.plot(splits, np.repeat(SSMSP_mae, K), label="SSMSP Univariate Avg")
ax.plot(splits, np.repeat(SSMSP_SNR_mae, K), label="SSMSP with SNR Avg")
ax.plot(splits, np.repeat(MSMSP_mae, K), label="MSMSP Univariate Avg")
ax.plot(splits, np.repeat(MSMSP_SNR_mae, K), label="MSMSP with SNR Avg")

ax.set_title("Mean Absolute Error over Multiple Train/Test Splits", fontsize=20)
ax.set_xlabel("Split", fontsize=14)
ax.set_ylabel("MAE", fontsize=14)
fig.savefig("Images/Univariate_and_SNR_MAE_Balanced.png", dpi=300)

fig, ax = plt.subplots()
ax.plot(splits, SSMSP["average_bias"], label="SSMSP Univariate")
ax.plot(splits, SSMSP_SNR["average_bias"], label="SSMSP with SNR")
ax.plot(splits, MSMSP["average_bias"], label="MSMSP Univariate")
ax.plot(splits, MSMSP_SNR["average_bias"], label="MSMSP with SNR")

ax.plot(splits, np.repeat(SSMSP_bias, K), label="SSMSP Univariate Avg")
ax.plot(splits, np.repeat(SSMSP_SNR_bias, K), label="SSMSP with SNR Avg")
ax.plot(splits, np.repeat(MSMSP_bias, K), label="MSMSP Univariate Avg")
ax.plot(splits, np.repeat(MSMSP_SNR_bias, K), label="MSMSP with SNR Avg")

ax.set_title("Model Bias over Multiple Train/Test Splits", fontsize=20)
ax.set_xlabel("Split", fontsize=14)
ax.set_ylabel("Bias", fontsize=14)
fig.savefig("Images/Univariate_and_SNR_bias_Balanced.png", dpi=300)