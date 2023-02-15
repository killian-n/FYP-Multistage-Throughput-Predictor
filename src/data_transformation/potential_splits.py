import csv
import pandas as pd
import numpy as np
import configparser
import sys

config = configparser.ConfigParser()
config.read('.env')
project_path = config['global']['PROJECT_PATH']
sys.path.append(project_path)

from src.data_transformation.preprocessor import DataPreProcessor

def generate_potential_splits(infile="", outfile="", max_shuffle=10000, train_percent=0.8):
    data = pd.read_csv(infile, index_col=None)
    data.columns = ["session", "low", "medium", "high"]

    all_train_traces = []
    all_test_traces = []
    all_train_low = []
    all_test_low = []
    all_train_medium = []
    all_test_medium = []
    all_train_high = []
    all_test_high = []

    for i in range(max_shuffle):
        train = data.sample(frac=train_percent)
        test = data.drop(train.index)
        train_traces = train["session"].tolist()    
        test_traces = test["session"].tolist()

        train_low = train["low"].sum()/data["low"].sum()
        train_medium = train["medium"].sum()/data["medium"].sum()
        train_high = train["high"].sum()/data["high"].sum()

        test_low = test["low"].sum()/data["low"].sum()
        test_medium = test["medium"].sum()/data["medium"].sum()
        test_high = test["high"].sum()/data["high"].sum()

        all_train_traces.append(train_traces)
        all_train_low.append(train_low)
        all_train_medium.append(train_medium)
        all_train_high.append(train_high)

        all_test_traces.append(test_traces)
        all_test_low.append(test_low)
        all_test_medium.append(test_medium)
        all_test_high.append(test_high)

    out_df = pd.DataFrame()
    out_df["train_traces"] = pd.Series(all_train_traces)
    out_df["train_low"] = all_train_low
    out_df["train_medium"] = all_train_medium
    out_df["train_high"] = all_train_high
    out_df["test_traces"] = pd.Series(all_test_traces)
    out_df["test_low"] = all_test_low
    out_df["test_medium"] = all_test_medium
    out_df["test_high"] = all_test_high

    out_df["diff1"] = out_df["train_low"] - out_df["train_medium"]
    out_df["diff1"] = out_df["diff1"].abs()
    out_df["diff2"] = out_df["train_low"] - out_df["train_high"]
    out_df["diff2"] = out_df["diff2"].abs()
    out_df["diff3"] = out_df["train_medium"] - out_df["train_high"]
    out_df["diff3"] = out_df["diff3"].abs()
    out_df["distribution_difference"] = out_df["diff1"] + out_df["diff2"] + out_df["diff3"]
    out_df.drop(columns=["diff1", "diff2", "diff3"], inplace=True)
    out_df.to_csv(outfile, index=False)

if __name__ == "__main__":

    # REQUIRES CSV WRITER LINES IN PREPROCESSOR BALANCE_LABELS TO BE COMMENTED IN
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", index_col=None)
    history_horizon_pairs = [[10,5], [15, 5], [20, 5], [25, 5], [15, 10], [20, 10], [25, 10]]

    for k in history_horizon_pairs:
        history, horizon = k[0], k[1]
        for i in range(raw_data["session"].max()+1):
            print("Processing Session:", i)
            pre_processor = DataPreProcessor(raw_data[raw_data["session"]==i], manual_mode=True)
            data = pre_processor.impute_and_normalise(pre_processor.get_df())
            x, y = pre_processor.create_sequences(data, history, horizon)
            y_labels = pre_processor.create_labels(y)
            x_balanced, y_balanced = pre_processor.balance_labels(x, y_labels)
        generate_potential_splits("Datasets/train_test_analysis/sequence_balance_h{}h{}.csv".format(history, horizon), outfile="Datasets/train_test_analysis/train_test_splits_h{}h{}.csv".format(history, horizon))