import numpy as np
import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import random
pd.options.mode.chained_assignment = None


class DataPreProcessor:
    def __init__(self, dataframe, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="univarte_scaler.sav"):
        
        # Working dataframes
        self.__df = dataframe[include_features+predict+["Timestamp", "session", "movement_type"]]
        self.__train = None
        self.__test = None

        # Predict variables
        self.__predict = predict
        self.__use_predict = use_predict

        # Numeric features
        self.__numeric_features = ["SNR", "CQI", "RSSI", "NRxRSRP", \
            "NRxRSRQ", "ServingCell_Distance", "Speed", "RSRQ", "RSRP", "DL_bitrate", "UL_bitrate"]
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
        self.__df = self.__df.replace({"-": np.nan})
        self.__df = self.__df.replace({"": np.nan})
        try:
            self.__df["Timestamp"] = pd.to_datetime(self.__df["Timestamp"],
             format="%Y.%m.%d_%H.%M.%S")
        except Exception as e:
            pass
        self.__df[self.__numeric_features+self.__geo_features] \
            = self.__df[self.__numeric_features+self.__geo_features].astype("float32")
        
        # Geographic features imputed using foward fill
        if self.__geo_features:
            self.__df[self.__geo_features] = self.__df[self.__geo_features].fillna(method="ffill")
        # Categorical features imputed using most frequent
        if self.__categorical_features:
            self.__df[self.__categorical_features].fillna(self.__df[self.__categorical_features].mode().iloc[0], inplace=True)

        # Output variables for use in models
        self.__scaler = scaler
        self.__scaler_file_name = scaler_file_name
        self.__x_train= []
        self.__y_train = []
        self.__x_test = []
        self.__y_test = []

        if not manual_mode:
            self.train_test_split()
            if self.__categorical_features:
                self.one_hot_encode()
            self.__train = self.impute_and_normalise(dataframe=self.__train)
            self.__test = self.impute_and_normalise(dataframe=self.__test, test=True, scaler=self.__scaler)
            self.__train = self.create_averaged_features(dataframe=self.__train)
            self.__test = self.create_averaged_features(dataframe=self.__test)
            self.__x_train, self.__y_train = self.create_sequences(dataframe=self.__train)
            self.__x_test, self.__y_test = self.create_sequences(dataframe=self.__test)
            self.save_scaler()

    def train_test_split(self, train_prop=0.8):
        no_of_traces = self.__df["session"].max()+1
        train_size = round(no_of_traces*train_prop)
        sample = random.sample(range(no_of_traces), train_size)
        train = pd.DataFrame()
        for i in sample:
            trace = self.__df[self.__df["session"]==i]
            train = pd.concat([train,trace], axis=0)
        self.__train = train
        self.__test = self.__df.drop(self.__train.index)

    # TODO : Handle the situation where train and test have different categorical features,
    #        as sequences must be consistent in length and column position
    def one_hot_encode(self):
        categorical_features = self.__categorical_features+["Timestamp"]
        cols = [i for i in self.__df.columns if i not in categorical_features]
        self.__df[cols] = self.__df[cols].astype("float32")
        self.__df = pd.get_dummies(self.__df)
        self.__categorical_features = list(set(self.__df.drop(columns=["Timestamp"])).difference(\
            self.__df[self.__numeric_columns+self.__geo_columns]))
    
    def impute_and_normalise(self, dataframe, test=False, scaler=None, return_true_values=False):
        if not scaler:
            if test:
                print("Error, test data must use a prior scaler created on the training data.")
                return None
            scaler = StandardScaler()
        # Numeric features imputed using KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        x_data = dataframe[self.__numeric_features+self.__geo_features+self.__categorical_features]
        normalised_x_data = scaler.fit_transform(x_data)
        new_values = imputer.fit_transform(normalised_x_data)
        if return_true_values:
            new_values = scaler.inverse_transform(new_values)
        dataframe[self.__numeric_features] = new_values
        dataframe[self.__numeric_features].astype("float32")
        if not test:
            self.__scaler = scaler
        return dataframe

    def create_averaged_features(self, dataframe, history_window=5):
        potential_features = ["SNR", "CQI", "RSSI", "NRxRSRQ", "NRxRSRP", "RSRP", "RSRQ"]
        features_to_average = [i for i in potential_features if i in self.__numeric_features and i in potential_features]
        updated_features = pd.DataFrame()
        groups = dataframe.groupby("session")
        for name, group in groups:
            group[features_to_average] = group[features_to_average].ewm(span=history_window, adjust=True).mean()
            updated_features = pd.concat([updated_features, group[features_to_average]], axis=0)
        dataframe[features_to_average] = updated_features
        return dataframe
    
    def create_sequences(self, dataframe=pd.DataFrame(), history_length=10, horizon_length=5):
        x = []
        y = []
        if dataframe.empty:
            dataframe = self.__df.copy()
        dataframe = dataframe.drop(columns=["movement_type", "Timestamp"])
        try:
            groups = dataframe.groupby("session")
        except Exception as e:
            dataframe["session"] = 0
            groups = dataframe.groupby("session")
        numeric_features = self.__numeric_features
        # Checking if feature is used to predict itself
        if self.__use_predict:
            numeric_features = self.__numeric_features[:-1]
        new_order = self.__predict + numeric_features + self.__geo_features + self.__categorical_features
        for name, group in groups:
            x_sequences = []
            y_sequences = []
            group = group.drop(columns=["session"])
            data = group.reindex(columns=new_order)
            for i in range(history_length, len(data)-horizon_length+1):
                x_sequences.append(data[i-history_length:i])
                y_sequences.append(data[self.__predict][i:i+horizon_length])
            x = x + x_sequences
            y = y + y_sequences
        return x, y

    def get_test_sequences(self):
        return self.__x_test, self.__y_test

    def get_train_sequences(self):
        return self.__x_train, self.__y_train

    def save_scaler(self, filename=None):
        if not filename:
            filename = self.__scaler_file_name
        filepath = "src/saved.objects/"+filename
        pickle.dump(self.__scaler, open(filepath, "wb"))



if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", index_col=None)
    pre_processor = DataPreProcessor(raw_data, manual_mode=False)
    x, y = pre_processor.get_test_sequences()
    print(x[0], y[0])
