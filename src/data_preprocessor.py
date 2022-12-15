import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


class WirelessDataPreProcessor:
    def __init__(self, dataframe, is_complete=True):
        self.__df = dataframe
        self.no_of_features = len(self.__df.columns)
        self.no_of_examples = self.__df.size
        self.numeric_columns = ["SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ", "ServingCell_Distance"]
        self.categorical_columns = list(set(self.__df.columns).difference(self.numeric_columns))
        self.group_columns = ["movement_type", "session"]
        self.geo_columns = ["ServingCell_Lon", "ServingCell_Lat"]
        self.interest_columns = ["DL_bitrate", "UL_bitrate"]

        self.__df = self.__df.replace({"-": np.nan})
        self.__df["Timestamp"] = pd.to_datetime(self.__df["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
        self.__df[["RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ", "ServingCell_Distance",
                  "ServingCell_Lat", "ServingCell_Lon", "DL_bitrate", "UL_bitrate"]] \
            = self.__df[["RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ", "ServingCell_Distance",
                        "ServingCell_Lat", "ServingCell_Lon", "DL_bitrate", "UL_bitrate"]].astype(float)

        if is_complete:
            self.__impute_grouped_missing_data()
            self.__create_grouped_history_column()
        else:
            self.__df = self.__impute_missing_data(self.__df)
            self.__df = self.__create_history_column(self.__df)

    def __impute_grouped_missing_data(self):
        groups = self.__df.groupby("movement_type")
        new_df = pd.DataFrame()
        for name, group in groups:
            imputed_df = self.__impute_missing_data(group)
            new_df = pd.concat([new_df, imputed_df], axis=0)
        self.__df = new_df

    def __create_grouped_history_column(self):
        groups = self.__df.groupby("session")
        new_df = pd.DataFrame()
        for name, group in groups:
            altered_data = self.__create_history_column(group)
            new_df = pd.concat([new_df, altered_data], axis=0)
        self.__df = new_df

    def __impute_missing_data(self, dataframe):
        dataframe[self.geo_columns].ffill(inplace=True)
        scaler = StandardScaler()
        x_data = dataframe[self.numeric_columns]
        normalized_x_data = scaler.fit_transform(x_data)
        imputer = KNNImputer(n_neighbors=5)
        imputed_normalized_array = imputer.fit_transform(normalized_x_data)
        imputed_array = scaler.inverse_transform(imputed_normalized_array)
        dataframe[self.numeric_columns] = imputed_array
        return dataframe

    def __create_history_column(self, dataframe, no_of_seconds=10):
        for column in self.interest_columns:
            new_column = "Average_" + column
            no_of_columns = len(dataframe.columns)
            history_column = []
            base_column = dataframe[column]
            # This loop creates history values for the first k rows,
            # assuming that values before the recorded time were zero.
            for i in range(no_of_seconds):
                history_value = base_column.iloc[:i + 1].sum() / no_of_seconds
                history_column.append(history_value)

            for i in range(no_of_seconds + 1, len(dataframe) + 1):
                history_value = base_column.iloc[i - no_of_seconds:i].mean()
                history_column.append(history_value)
            dataframe[new_column] = history_column
        return dataframe

    def get_df(self):
        return self.__df
