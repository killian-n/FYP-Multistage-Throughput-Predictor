import numpy as np
import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


class DataPreProcessor:
    def __init__(self, dataframe, include_features=["SNR", "CQI", "UL_bitrate"], predict=["DL_bitrate"], use_predict=True, manual_mode=False):
        self.__df = dataframe[include_features+predict+["Timestamp", "session", "movement_type"]]

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
            = self.__df[self.__numeric_features+self.__geo_features].astype(float)

        # Output variables for use in models
        self.__scaler = None
        self.__x_sequences = []
        self.__y_sequences = []

        if not manual_mode:
            self.impute_missing_data()
            self.create_averaged_features()
            if self.__categorical_features:
                self.one_hot_encode()
            self.normalise()
    
    def impute_missing_data(self):
        # Geographic features imputed using foward fill
        if self.__geo_features:
            self.__df[self.__geo_features] = self.__df[self.__geo_features].fillna(method="ffill")
        # Categorical features imputed using most frequent
        if self.__categorical_features:
            self.__df[self.__categorical_features].fillna(self.__df[self.__categorical_features].mode().iloc[0], inplace=True)
        # Numeric features imputed using KNNImputer
        scaler = StandardScaler()
        imputer = KNNImputer(n_neighbors=5)
        x_data = self.__df[self.__numeric_features]
        normalised_x_data = scaler.fit_transform(x_data)
        imputed_normalised_array = imputer.fit_transform(normalised_x_data)
        imputed_array = scaler.inverse_transform(imputed_normalised_array)
        self.__df[self.__numeric_factors] = imputed_array


    def create_averaged_features(self, history_window=5):
        potential_features = ["SNR", "CQI", "RSSI", "NRxRSRQ", "NRxRSRP", "RSRP", "RSRQ"]
        features_to_average = [i for i in potential_features if i in self.__numeric_features and i in potential_features]
        updated_features = pd.DataFrame()
        groups = self.__df.groupby("session")
        for name, group in groups:
            group[features_to_average] = group[features_to_average].ewm(span=history_window, adjust=True).mean()
            updated_features = pd.concat([updated_features, group[features_to_average]], axis=0)
        self.__df[features_to_average] = updated_features
    
    def normalise(self):
        scaler = StandardScaler()
        if self.__geo_features:
            x_data = self.__df[]




if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", index_col=None)
    pre_processor = DataPreProcessor(raw_data, manual_mode=False)