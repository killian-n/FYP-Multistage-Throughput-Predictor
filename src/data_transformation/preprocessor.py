import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import random
import json
from prettytable import PrettyTable
import sys
pd.options.mode.chained_assignment = None
import configparser
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)


class DataPreProcessor:
    def __init__(self, dataframe, include_features=[], predict=["DL_bitrate"], use_predict=True,
                  manual_mode=False, scaler=None, scaler_file_name="univariate_scaler.sav",
                    scale_data=False, history=10, horizon=5, create_train_test_split=False, name="univariate", remove_idle_periods=3):
        # Metadata
        metadata = ["Timestamp", "session", "movement_type"]
        self._name = name
        self.__history_length = history
        self.__horizon_length = horizon
        self.__create_train_test_split = create_train_test_split
        self.__scale_data = scale_data
        self.__imputer = None
        # Idle periods
        self.__df = dataframe
        self.__df["State"] = self.__df["State"].replace({"D": 1, "I": 0})
        if remove_idle_periods:
            self.remove_idle_periods(min_period=remove_idle_periods)   
        # Working dataframes
        self.__df = self.__df[include_features+predict+metadata]
        self.__train = None
        self.__test = None

        # Predict variables
        self.__predict = predict
        self.__use_predict = use_predict

        # Numeric features
        self.__numeric_features = ["SNR", "CQI", "RSSI", "NRxRSRP", \
            "NRxRSRQ", "ServingCell_Distance", "Speed", "RSRQ", "RSRP", "DL_bitrate", "UL_bitrate", "State"]
        self.__numeric_features = [i for i in self.__numeric_features if i in include_features and i in self.__numeric_features]
        if use_predict:
            for i in predict:
                if i not in self.__numeric_features:
                    self.__numeric_features.append(i)
        # Geological features
        self.__geo_features = ["Longitude", "Latitude", "ServingCell_Lon", "ServingCell_Lat"]
        self.__geo_features = [i for i in self.__geo_features if i in self.__geo_features and i in include_features]

        # Categorical features
        self.__categorical_features = list(set(self.__df.drop(columns=["Timestamp"])).difference(\
            self.__df[self.__numeric_features+self.__geo_features]))
        self.__categorical_features = [i for i in self.__categorical_features if i in self.__categorical_features and i in include_features]

        # Basic formatting
        self.__df = self.__df.replace({"-": np.nan, "": np.nan})
        try:
            self.__df["Timestamp"] = pd.to_datetime(self.__df["Timestamp"],
             format="%Y.%m.%d_%H.%M.%S")
        except Exception as e:
            pass
        # Potential memory issue with float64
        self.__df[self.__numeric_features+self.__geo_features] \
            = self.__df[self.__numeric_features+self.__geo_features].astype("float32")
        if self.__use_predict:
            numeric_features = self.__numeric_features[:-(len(self.__predict))]
        new_order = self.__predict + numeric_features + self.__geo_features + self.__categorical_features + metadata
        self.__df = self.__df[new_order]
        
        # Geographic features imputed using foward fill
        if self.__geo_features:
            self.__df[self.__geo_features] = self.__df[self.__geo_features].fillna(method="ffill")
        # Categorical features imputed using most frequent
        if self.__categorical_features:
            self.__df[self.__categorical_features].fillna(self.__df[self.__categorical_features].mode().iloc[0], inplace=True)

        if self.__use_predict:
            self.__features_to_scale = self.__predict+self.__numeric_features[:-(len(self.__predict))]+self.__geo_features
        else:
            self.__features_to_scale = self.__numeric_features+self.__geo_features


        # Remove a large outlier that messed with scaler for this feature
        if "NRxRSRQ" in self.__numeric_features:
            row = self.__df[self.__df["NRxRSRQ"]==self.__df["NRxRSRQ"].min()].index[0]
            column = self.__df.columns.get_loc("NRxRSRQ")
            print("ROw", row)
            print("column", column)
            print(self.__df.iloc[row, column])
            self.__df.iloc[row, column] = 0
            self.__df.iloc[row, column] = self.__df["NRxRSRQ"].min()
        
        # Output variables for use in models
        self.__features = []

        self.__scaler = scaler
        self.__scaler_length = 0
        self.__scaler_file_name = name+"_scaler.sav"
        self.__x_train= []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []
        # For Multi-stage Models
        # Train
        self.__x_train_low = []
        self.__y_train_low = []
        self.__x_train_medium = []
        self.__y_train_medium = []
        self.__x_train_high = []
        self.__y_train_high = []
        # Test
        self.__x_test_low = []
        self.__y_test_low = []
        self.__x_test_medium = []
        self.__y_test_medium = []
        self.__x_test_high = []
        self.__y_test_high = []

        # Labels for basic predictor model
        self.__label_dict = {"low": (1,0,0), "medium": (0,1,0), "high": (0,0,1)}
        self.__sparse_label_dict = {"low":0, "medium": 1, "high": 2}
        self.__class_weights = {}

        # Train
        self.__y_train_labels = []
        self.__y_train_labels_sparse = []
        # Sequence, targets after balancing
        self.__x_train_balanced = []
        self.__y_train_balanced = []

        # Test
        self.__y_test_labels = []
        self.__y_test_labels_sparse = []

        if not manual_mode:
            return_unscaled = not self.__scale_data
            self.preprocess(return_unscaled=return_unscaled)
    
    def preprocess(self, return_unscaled=False):
        self.one_hot_encode()
        self.train_test_split()
        self.preprocess_numeric(train_df=self.__train, test_df=self.__test, return_unscaled=return_unscaled)

    def preprocess_numeric(self, train_df=None, test_df=None, return_unscaled=False):
            self.__train = train_df
            self.__test = test_df
            # MinMaxScaler so it doesnt matter if we scale before impute and we need to scale before impute of KNNImputer
            self.__train = self.apply_scaler(self.__train,train=True)
            self.__train = self.impute(dataframe=self.__train, train=True)
            self.__test =self.apply_scaler(self.__test)
            self.__test = self.impute(dataframe=self.__test)
            if return_unscaled:
                self.__train = self.inverse_scale(self.__train, is_x=True)
                self.__test = self.inverse_scale(self.__test, is_x=True)
            self.__train = self.create_averaged_features(dataframe=self.__train)
            self.__test = self.create_averaged_features(dataframe=self.__test)
            self.__x_train, self.__y_train = self.create_sequences(self.__train, self.__history_length, self.__horizon_length)
            self.__x_test, self.__y_test = self.create_sequences(self.__test, self.__history_length, self.__horizon_length)
            scaled = not return_unscaled
            self.__y_train_labels = self.create_labels(self.__y_train, sparse=False, scaled=scaled)
            self.__y_test_labels = self.create_labels(self.__y_test, sparse=False, scaled=scaled)
            self.__y_train_labels_sparse = self.create_labels(self.__y_train, sparse=True, scaled=scaled)
            self.__y_test_labels_sparse = self.create_labels(self.__y_test, sparse=True, scaled=scaled)
            self.__x_train_balanced, self.__y_train_labels_balanced = self.balance_labels(self.__x_train, self.__y_train_labels)
            self.__y_train_balanced, self.__y_train_labels_sparse_balanced = self.balance_labels(self.__y_train, self.__y_train_labels_sparse, sparse=True)
            self.separate_by_label(train=True)
            self.separate_by_label(train=False)
            self.save_scaler()
            self.save_imputer()
            self.save_datasets()
            self.save_class_weights()

    def train_test_split(self, train_prop=0.8):
        try:
            potential_splits = pd.read_csv(config["global"]["PROJECT_PATH"]+"/Datasets/train_test_analysis/train_test_splits_h{}h{}.csv".format(self.__history_length, self.__horizon_length), index_col=None)
        except Exception as e:
            for i in range(self.__df["session"].max()+1):
                print("Processing Session:", i)
                pre_processor = DataPreProcessor(self.__df[self.__df["session"]==i], manual_mode=True, create_train_test_split=True)
                data = pre_processor.impute(pre_processor.get_df())
                x, y = pre_processor.create_sequences(data, self.__history_length, self.__horizon_length)
                y_labels = pre_processor.create_labels(y)
                x_balanced, y_balanced = pre_processor.balance_labels(x, y_labels)
            self.__generate_potential_splits(config["global"]["PROJECT_PATH"]+"/Datasets/train_test_analysis/sequence_balance_h{}h{}.csv".format(self.__history_length, self.__horizon_length),
             outfile=config["global"]["PROJECT_PATH"]+"/Datasets/train_test_analysis/train_test_splits_h{}h{}.csv".format(self.__history_length, self.__horizon_length), train_prop=train_prop)
            self.train_test_split(train_prop=train_prop)

        potential_splits = potential_splits.sort_values(by="distribution_difference")
        selected_train_test_split = potential_splits.iloc[0]
        train_traces = json.loads(selected_train_test_split["train_traces"])
        test_traces = json.loads(selected_train_test_split["test_traces"])
        self.__train = self.__df[self.__df["session"].isin(train_traces)]
        self.__test = self.__df[self.__df["session"].isin(test_traces)]
        table = PrettyTable()
        table.field_names = potential_splits.drop(columns=["train_traces", "test_traces"]).columns.to_list()
        table.add_row(selected_train_test_split.drop(labels=["train_traces", "test_traces"]).to_list())
        print("This is the selected Train/Test Split:")
        print(table)
            
    def __generate_potential_splits(self, infile="", outfile="", max_shuffle=10000, train_prop=0.8):
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
            train = data.sample(frac=train_prop)
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

    def one_hot_encode(self):
        self.__df = pd.get_dummies(self.__df.drop(columns=["movement_type"]))
        self.__categorical_features = list(set(self.__df.drop(columns=["Timestamp", "session"])).difference(\
            self.__df[self.__numeric_features+self.__geo_features]))
        print("Categorical features:", self.__categorical_features)
    
    def impute(self, dataframe, train=False):
        # IMPORTANT: See Data_Expolration notebook for reasoning on imputation choices

        # IMPORTANT IMPORTANT:: IF we ever get around to trying to implement this model in practice then: The MULTISTAGE MODEL WILL HAVE TO STORE MIN VALUES FOR ALL THE FEATURES ASWELL AS THE IMPUTER

        if "CQI" in self.__numeric_features:
            dataframe["CQI"].fillna(-1, inplace=True)
        
        if "SNR" in self.__numeric_features:
            dataframe["SNR"].fillna(dataframe["SNR"].min()-1, inplace=True)

        if "RSSI" in self.__numeric_features:
            dataframe["RSSI"].fillna(self.__train["RSSI"].min()-1, inplace=True)
        # NRxRSRP and NRxRSRQ require KNN-Imputation.
        require_KNN = ["NRxRSRP", "NRxRSRQ", "RSRP", "RSRQ"]
        require_KNN = [i for i in require_KNN if i in self.__numeric_features]
        if require_KNN:
            data_to_impute = dataframe.drop(columns=["Timestamp", "session"]).columns.tolist()
            if train:
                self.__imputer = KNNImputer(n_neighbors=5)
                dataframe[data_to_impute] = self.__imputer.fit_transform(dataframe[data_to_impute])
            else:
                dataframe[data_to_impute] = self.__imputer.transform(dataframe[data_to_impute])
        return dataframe
    
    def knn_impute(self, dataframe, train=False):
        # NRxRSRP and NRxRSRQ require KNN-Imputation.
        require_KNN = ["NRxRSRP", "NRxRSRQ", "RSRP", "RSRQ", "CQI", "SNR", "RSSI"]
        require_KNN = [i for i in require_KNN if i in self.__numeric_features]
        if require_KNN:
            data_to_impute = dataframe.drop(columns=["Timestamp", "session"]).columns.tolist()
            if train:
                self.__imputer = KNNImputer(n_neighbors=5)
                dataframe[data_to_impute] = self.__imputer.fit_transform(dataframe[data_to_impute])
            else:
                dataframe[data_to_impute] = self.__imputer.transform(dataframe[data_to_impute])
        return dataframe
        

    def apply_scaler(self, dataframe, train=False):
        # isolate numeric features
        if self.__use_predict:
            self.__features_to_scale = self.__predict+self.__numeric_features[:-(len(self.__predict))]+self.__geo_features+self.__categorical_features
        else:
            self.__features_to_scale = self.__numeric_features+self.__geo_features+self.__categorical_features
        data_to_scale = dataframe[self.__features_to_scale]
        if train:
            self.__scaler = MinMaxScaler((0, 1))
            scaled_data = self.__scaler.fit_transform(data_to_scale)
            self.__scaler_length = self.__scaler.n_features_in_
        else:
            scaled_data = self.__scaler.transform(data_to_scale)
        dataframe[self.__features_to_scale] = scaled_data
        return dataframe

    def remove_idle_periods(self, min_period=3):
        # create a idle period count column
        idle_time = self.__df.groupby(self.__df['State'].diff().ne(0).cumsum()).cumcount()
        self.__df["idle"] = idle_time+1
        # remove counts for download as its unrequired
        self.__df.loc[(self.__df["idle"]>0) & (self.__df["State"]==1), "idle"] = 0
        # obtain the timestamps where the idle period was greater than the minimum
        idle_periods = self.__df[self.__df["idle"]>0]["idle"]
        # extract the indices
        idle_indices = idle_periods.index
        idle_periods = idle_periods.to_list()
        i = 0
        while i < len(idle_periods):
            idle_len = idle_periods[i]
            if i < len(idle_periods)-1:
                next_val = idle_periods[i+1]
            else:
                next_val = idle_periods[i]
            if idle_len == 1:
                i += 1
            elif next_val > idle_len:
                i += 1
            elif idle_len >= min_period:
                cur_index = idle_indices[i]
                period_to_remove = list(range((cur_index-idle_len)+1, cur_index+1))
                self.__df.drop(period_to_remove, inplace=True)
                i+=1
            else:
                i+=1
        self.__df.reset_index(inplace=True)

    def inverse_scale(self, dataframe, is_x=True):
        # isolate numeric features
        input_array = dataframe[self.__features_to_scale].to_numpy()
        input_shape = input_array.shape
        if is_x: 
                input_array = self.__scaler.inverse_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self.__scaler.n_features_in_))
            transform[:,0] = input_array
            transform = self.__scaler.inverse_transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        dataframe[self.__features_to_scale] = input_array
        return dataframe
    
    def create_averaged_features(self, dataframe, history_window=5):
        potential_features = ["SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ", "RSRQ", "RSRP"]
        features_to_average = [i for i in potential_features if i in self.__numeric_features and i in potential_features]
        updated_features = pd.DataFrame()
        groups = dataframe.groupby("session")
        for name, group in groups:
            group[features_to_average] = group[features_to_average].ewm(span=history_window, adjust=True).mean()
            updated_features = pd.concat([updated_features, group[features_to_average]], axis=0)
        dataframe[features_to_average] = updated_features
        return dataframe
    
    def create_sequences(self, dataframe, history_length, horizon_length):
        self.__history_length = history_length
        self.__horizon_length = horizon_length
        # x, y are lists of dataframes
        x = []
        y = []
        #create copy
        if dataframe.empty:
            dataframe = self.__df.copy()
        # drop metadata
        dataframe = dataframe.drop(columns=["Timestamp"])
        # group by traces
        try:
            groups = dataframe.groupby("session")
        # if only 1 trace was provided it may not include the session column
        except Exception as e:
            dataframe["session"] = 0
            groups = dataframe.groupby("session")
        numeric_features = self.__numeric_features
        # Checking if feature is used to predict itself
        if self.__use_predict:
            numeric_features = self.__numeric_features[:-(len(self.__predict))]
        # ordering the columns
        new_order = self.__predict + numeric_features + self.__geo_features + self.__categorical_features
        for name, group in groups:
            x_sequences = []
            y_sequences = []
            group = group.drop(columns=["session"])
            data = group[new_order]
            self.__features = list(data.columns)
            for i in range(self.__history_length, len(data)-self.__horizon_length+1):
                x_sequences.append(data[i-self.__history_length:i])
                y_sequences.append(data[self.__predict][i:i+self.__horizon_length])
            x = x + x_sequences
            y = y + y_sequences
        return x, y

    def create_labels(self, y_sequences=[], scaled=True, sparse=False):
        y_labels = []
        y_sequences = np.array(y_sequences)
        if sparse:
            label_dict = self.__sparse_label_dict
        else:
            label_dict = self.__label_dict
        for sequence in y_sequences:
            # adding columns to the np.array to match the shape used in the scaler.
            diff = self.__scaler_length - sequence.shape[1]
            if diff > 0:
                transform = np.zeros((sequence.shape[0],self.__scaler_length))
                transform[:,:-diff] = sequence
            else: 
                transform = sequence
            if scaled:
                transform = self.__scaler.inverse_transform(transform)
            average_throughput = (sum(transform[:,0])/len(transform[:,0]))/1000
            if average_throughput < 1:
                y_labels.append(label_dict["low"])
            elif average_throughput > 5:
                y_labels.append(label_dict["high"])
            else:
                y_labels.append(label_dict["medium"])
        return y_labels

    # KEEP FOR CREATE_TRAIN_TEST
    def balance_labels(self, x_sequences, labels, sparse=False, upsample=False):
        # PROBABLY DONT NEED TO BALANCE TEST SET
        low_x = []
        medium_x = []
        high_x = []
        low_y = []
        medium_y = []
        high_y = []

        if sparse:
            for sequence, target in zip(x_sequences, labels):
                if target == 0:
                    low_x.append(sequence)
                    low_y.append(target)
                elif target == 1:
                    medium_x.append(sequence)
                    medium_y.append(target)
                else:
                    high_x.append(sequence)
                    high_y.append(target)
        else:
            for sequence, target in zip(x_sequences, labels):
                if target[0] == 1:
                    low_x.append(sequence)
                    low_y.append(target)
                elif target[1] == 1:
                    medium_x.append(sequence)
                    medium_y.append(target)
                else:
                    high_x.append(sequence)
                    high_y.append(target)

        if self.__create_train_test_split:
            with open(config["global"]["PROJECT_PATH"]+"Datasets/train_test_analysis/sequence_balance_h{}h{}.csv".format(self.__history_length, self.__horizon_length), "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.__df["session"].max(), len(low_x), len(medium_x), len(high_x)])
        
        # Upsampling
        if upsample:
            # High
            len_high = len(high_y)
            len_medium = len(medium_y)
            len_low = len(low_y)

            if len_high >= len_medium and len_high >= len_low:
                sample_low_indices = resample(list(range(0,len_low)),
                replace=True,
                n_samples=len_high-len_low,
                random_state=13)
                
                sample_medium_indices = resample(list(range(0,len_medium)),
                replace=True,
                n_samples=len_high-len_medium,
                random_state=13)
                
                for i in sample_low_indices:
                    low_x.append(low_x[i])
                    low_y.append(low_y[i])
                for i in sample_medium_indices:
                    medium_x.append(medium_x[i])
                    medium_y.append(medium_y[i])
            # Low
            elif len_low >= len_medium and len_low >= len_high:

                sample_high_indices = resample(list(range(0,len_high)),
                replace=True,
                n_samples=len_low-len_high,
                random_state=13)
                
                sample_medium_indices = resample(list(range(0,len_medium)),
                replace=True,
                n_samples=len_low-len_medium,
                random_state=13)
                
                for i in sample_high_indices:
                    high_x.append(high_x[i])
                    high_y.append(high_y[i])
                for i in sample_medium_indices:
                    medium_x.append(medium_x[i])
                    medium_y.append(medium_y[i])
            # Medium
            else:
                sample_high_indices = resample(list(range(0,len_high)),
                replace=True,
                n_samples=len_medium-len_high,
                random_state=13)
                
                sample_low_indices = resample(list(range(0,len_low)),
                replace=True,
                n_samples=len_medium-len_low,
                random_state=13)
                
                for i in sample_high_indices:
                    high_x.append(high_x[i])
                    high_y.append(high_y[i])
                for i in sample_low_indices:
                    low_x.append(low_x[i])
                    low_y.append(low_y[i])

        if not upsample:
            pass
            #minimum = min(len(low_y), len(medium_y), len(high_y))
            # low_x = low_x[:minimum]
            # low_y = low_y[:minimum]
            # medium_x = medium_x[:minimum]
            # medium_y = medium_y[:minimum]
            # high_x = high_x[:minimum]
            # high_y = high_y[:minimum]
            # print("Low", len(low_x), "Med", len(medium_x), "High", len(high_x))

        x = low_x + medium_x + high_x
        y = low_y + medium_y + high_y
        return x, y

    def separate_by_label(self, train=True):
        low_x = []
        medium_x = []
        high_x = []
        low_y = []
        medium_y = []
        high_y = []
        reverse_label_dict = {}

        for key in self.__label_dict:
            val = self.__label_dict[key]
            reverse_label_dict[val] = key

        if train:
            x_sequences = self.__x_train
            y_sequences = self.__y_train
            labels = self.__y_train_labels
        else:
            x_sequences = self.__x_test
            y_sequences = self.__y_test
            labels = self.__y_test_labels

        for i in range(len(labels)):
            if reverse_label_dict[labels[i]] == "low":
                low_x.append(x_sequences[i])
                low_y.append(y_sequences[i])
            elif reverse_label_dict[labels[i]] == "medium":
                medium_x.append(x_sequences[i])
                medium_y.append(y_sequences[i])
            else:
                high_x.append(x_sequences[i])
                high_y.append(y_sequences[i])

        if train:
            self.__x_train_low = low_x
            print("LOW x train", np.array(self.__x_train_low).shape)
            self.__y_train_low = low_y
            self.__x_train_medium = medium_x
            print("Medium x train", np.array(self.__x_train_medium).shape)
            self.__y_train_medium = medium_y
            self.__x_train_high = high_x
            print("High x train", np.array(self.__x_train_high).shape)
            self.__y_train_high = high_y
            self.__class_weights[self.__sparse_label_dict["low"]] = len(self.__y_train)/(3*len(self.__y_train_low))
            self.__class_weights[self.__sparse_label_dict["medium"]] = len(self.__y_train)/(3*len(self.__y_train_medium))
            self.__class_weights[self.__sparse_label_dict["high"]] = len(self.__y_train)/(3*len(self.__y_train_high))

        else:
            self.__x_test_low = low_x
            self.__y_test_low = low_y
            self.__x_test_medium = medium_x
            self.__y_test_medium = medium_y
            self.__x_test_high = high_x
            self.__y_test_high = high_y




    def get_test_sequences(self):
        return np.array(self.__x_test), np.array(self.__y_test)

    def get_train_sequences(self):
        return np.array(self.__x_train), np.array(self.__y_train)
    
    def get_balanced_train_sequences(self):
        return np.array(self.__x_train_balanced), np.array(self.__y_train_balanced)

    def get_label_predictor_train(self, sparse=False, upsampled=False):
        if sparse:
            if upsampled:
                return np.array(self.__x_train_balanced), np.array(self.__y_train_labels_sparse_balanced, ndmin=2)
            return np.array(self.__x_train), np.array(self.__y_train_labels_sparse, ndmin=2)
        if upsampled:
            np.array(self.__x_train_balanced), np.array(self.__y_train_labels_balanced, ndmin=2)
        return np.array(self.__x_train), np.array(self.__y_train_labels, ndmin=2)

    def get_label_predictor_test(self, sparse=False):
        if sparse:
            return np.array(self.__x_test), np.array(self.__y_test_labels_sparse, ndmin=2)
        return np.array(self.__x_test), np.array(self.__y_test_labels, ndmin=2)

    def get_low_train_sequences(self):
        return np.array(self.__x_train_low), np.array(self.__y_train_low)

    def get_medium_train_sequences(self):
        return np.array(self.__x_train_medium), np.array(self.__y_train_medium)

    def get_medium_test_sequences(self):
        return np.array(self.__x_test_medium), np.array(self.__y_test_medium)

    def get_high_train_sequences(self):
        return np.array(self.__x_train_high), np.array(self.__y_train_high)

    def get_low_test_sequences(self):
        return np.array(self.__x_test_low), np.array(self.__y_test_low)

    def get_high_test_sequences(self):
        return np.array(self.__x_test_high), np.array(self.__y_test_high)

    def get_label_dict(self):
        return self.__label_dict

    def get_history_length(self):
        return self.__history_length

    def get_horizon_length(self):
        return self.__horizon_length

    def get_feature_order(self):
        return self.__features

    def get_train(self):
        return self.__train
    
    def get_test(self):
        return self.__test

    def get_scaler(self):
        return self.__scaler

    def get_scaler_length(self):
        return self.__scaler_length

    def get_df(self):
        return self.__df

    def get_class_weights(self):
        return self.__class_weights
    
    def save_class_weights(self, filename=""):
        if not filename:
            filename = "{}_class_weights.sav".format(self._name)
        saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_objects_path[-1] not in ["\\", "/"]:
            saved_objects_path += "/"
        filepath = saved_objects_path+filename
        pickle.dump(self.__class_weights, open(filepath, "wb"))

    def set_train(self, train_df=pd.DataFrame()):
        self.__train = train_df

    def set_test(self, test_df=pd.DataFrame()):
        self.__test = test_df

    def save_datasets(self):
        # Training Datasets
        train_dir = config["global"]["TRAINING_DATASETS_PATH"]
        if train_dir[-1] not in ["\\", "/"]:
            train_dir += "/"
        x, y = self.get_train_sequences()
        np.save(train_dir+"{}_train_x".format(self._name), x)
        np.save(train_dir+"{}_train_y".format(self._name), y)
        x, y = self.get_low_train_sequences()
        np.save(train_dir+"{}_low_train_x".format(self._name), x)
        np.save(train_dir+"{}_low_train_y".format(self._name), y)
        x, y = self.get_medium_train_sequences()
        np.save(train_dir+"{}_medium_train_x".format(self._name), x)
        np.save(train_dir+"{}_medium_train_y".format(self._name), y)
        x, y = self.get_high_train_sequences()
        np.save(train_dir+"{}_high_train_x".format(self._name), x)
        np.save(train_dir+"{}_high_train_y".format(self._name), y)
        x, y = self.get_label_predictor_train(sparse=False)
        np.save(train_dir+"{}_classifier_train_x".format(self._name), x)
        np.save(train_dir+"{}_classifier_train_y".format(self._name), y)

        x, y = self.get_label_predictor_train(sparse=False, upsampled=True)
        np.save(train_dir+"{}_classifier_balanced_train_x".format(self._name), x)
        np.save(train_dir+"{}_classifier_balanced_train_y".format(self._name), y)
        x, y = self.get_balanced_train_sequences()
        np.save(train_dir+"{}_balanced_train_x".format(self._name), x)
        np.save(train_dir+"{}_balanced_train_y".format(self._name), y)
        
        # Testing Datasets
        test_dir = config["global"]["TESTING_DATASETS_PATH"]
        if test_dir[-1] not in ["\\", "/"]:
            test_dir += "/"

        x, y = self.get_test_sequences()
        np.save(test_dir+"{}_test_x".format(self._name), x)
        np.save(test_dir+"{}_test_y".format(self._name), y)
        x, y = self.get_low_test_sequences()
        np.save(test_dir+"{}_low_test_x".format(self._name), x)
        np.save(test_dir+"{}_low_test_y".format(self._name), y)
        x, y = self.get_medium_test_sequences()
        np.save(test_dir+"{}_medium_test_x".format(self._name), x)
        np.save(test_dir+"{}_medium_test_y".format(self._name), y)
        x, y = self.get_high_test_sequences()
        np.save(test_dir+"{}_high_test_x".format(self._name), x)
        np.save(test_dir+"{}_high_test_y".format(self._name), y)
        x, y = self.get_label_predictor_test()
        np.save(test_dir+"{}_classifier_test_x".format(self._name), x)
        np.save(test_dir+"{}_classifier_test_y".format(self._name), y)

    def save_scaler(self, filename=None):
        if not filename:
            filename = self.__scaler_file_name
        saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_objects_path[-1] not in ["\\", "/"]:
            saved_objects_path += "/"
        filepath = saved_objects_path+filename
        pickle.dump(self.__scaler, open(filepath, "wb"))

    def save_imputer(self, filename=None):
        if self.__imputer:
            if not filename:
                filename = self._name+"_imputer.sav"
            saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
            if saved_objects_path[-1] not in ["\\", "/"]:
                saved_objects_path += "/"
            filepath = saved_objects_path+filename
            pickle.dump(self.__imputer, open(filepath, "wb"))

    def is_scaled(self):
        return self.__scale_data

if __name__ == "__main__":
    raw_data = pd.read_csv(config["global"]["PROJECT_PATH"]+"/Datasets/Raw/all_4G_data.csv", index_col=None)
    preprocessor = DataPreProcessor(raw_data, include_features=["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "UL_bitrate",
                                                                 "State","NetworkMode", "Longitude", "Latitude", "NRxRSRQ", "NRxRSRP"], manual_mode=True)
    # preprocessor.one_hot_encode()
    # data = preprocessor.get_df()
    # data = preprocessor.apply_scaler(data, train=True)
    # data = preprocessor.knn_impute(data, train=True)
    # data = preprocessor.inverse_scale(data, is_x=True)
    # # data = preprocessor.create_averaged_features(dataframe=data)
    # data.to_csv(config["global"]["PROJECT_PATH"]+"/Datasets/unaveraged_processed_network_data.csv", index=False, encoding="utf-8")

    # data = np.random.sample((100, 5))
    # data = np.round(data, 0)
    # scaler = MinMaxScaler((0,1))
    # scaled = scaler.fit_transform(data)

    #print(data[0:5])
    #print("before scaling")
    #print(data[[0,3, 5]])


    #print("AFter scaling")
    #print(scaled[0])
