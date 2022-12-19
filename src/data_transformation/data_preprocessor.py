import numpy as np
import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

##################################################################
# DEPRECATED
# WILL BE REMOVING IN FUTURE COMMIT
#
# Alot of invalid assumptions including:
# 1. Assuming preprocessing takes place before train/test split.
# 2. Including features which are not reported by the device and as such would not appear in
#    actual real world data.
# 3. Preforming imputation on seperately, features such as movement_type will not be available for test data.
# 4. In future projects should preform feature selection before writing any preprocessing code.
# Rewriting preprocessing code in new file: "preprocessor.py"
##################################################################
class WirelessDataPreProcessor:
    def __init__(self, dataframe, is_grouped=True, manual_mode=False):

        # Initial class variables
        self.__df = dataframe
        self.__numeric_columns = ["SNR", "CQI", "RSSI", "NRxRSRP", \
            "NRxRSRQ", "ServingCell_Distance", "Speed", "RSRQ", "RSRP"]
        self.__geo_columns = ["Longitude", "Latitude", "ServingCell_Lon", "ServingCell_Lat"]
        self.__interest_columns = ["DL_bitrate", "UL_bitrate"]
        self.__historic_columns = ["Average_DL_bitrate", "Average_UL_bitrate"]

        self.__categorical_columns = list(set(self.__df.drop(columns=["Timestamp"])).difference(\
            self.__df[self.__numeric_columns+self.__geo_columns+self.__interest_columns]))

        # Basic formatting
        self.__df = self.__df.replace({"-": np.nan})
        self.__df = self.__df.replace({"": np.nan})
        try:
            self.__df["Timestamp"] = pd.to_datetime(self.__df["Timestamp"],
             format="%Y.%m.%d_%H.%M.%S")
        except Exception as e:
            pass
        self.__df[self.__numeric_columns+self.__interest_columns] \
            = self.__df[self.__numeric_columns+self.__interest_columns].astype(float)


        # Output variables for use in models
        self.__scaler = None
        self.__x_sequences = []
        self.__y_sequences = []

        # Options relating to the input shape of the dataframe
        if not manual_mode:
            if is_grouped:
                self.__categorical_columns.remove("session")
                self.impute_grouped_missing_data()
                self.create_grouped_history_column()
            else:
                self.__df = self.impute_missing_data(self.__df)
                self.__df = self.create_history_column(self.__df)
            self.one_hot_encode()
            self.scale_data()

    def impute_grouped_missing_data(self):
        '''
        Groups data by "movement_type" and uses function self.impute_missing_data() to 
        fill in missing data using KNN for each movement_type seperately.
        This is better than imputing over the entire column as the movement type plays a large
        role in the distribution of the data-points.

        Assigns updated dataframe to df.
        '''
        groups = self.__df.groupby("movement_type")
        new_df = pd.DataFrame()
        for name, group in groups:
            imputed_df = self.impute_missing_data(group)
            new_df = pd.concat([new_df, imputed_df], axis=0)
        self.__df = new_df

    def create_grouped_history_column(self):
        groups = self.__df.groupby("session")
        new_df = pd.DataFrame()
        for name, group in groups:
            altered_data = self.create_history_column(group)
            new_df = pd.concat([new_df, altered_data], axis=0)
        self.__df = new_df

    def impute_missing_data(self, dataframe):
        # Long, lat positions of the towers get imputed using foward fill
        dataframe[self.__geo_columns] = dataframe[self.__geo_columns].fillna(method="ffill")
        # Categorical features use most frequent
        dataframe[self.__categorical_columns] = dataframe[self.__categorical_columns]\
            .fillna(dataframe[self.__categorical_columns].mode().iloc[0])
        # One-hot-encode categororical data for KNN imputation
        categorical_features = self.__categorical_columns+["Timestamp", "session"]
        cols = [i for i in self.__df.columns if i not in categorical_features]
        dataframe[cols] = dataframe[cols].astype(float)
        dataframe = pd.get_dummies(dataframe)
        self.__categorical_columns = list(set(dataframe.drop(columns=["Timestamp"])).difference(\
            dataframe[self.__numeric_columns+self.__geo_columns+self.__interest_columns]))
        # Numeric features use the KNNImputer
        scaler = StandardScaler()
        x_data = dataframe[self.__numeric_columns+self.__interest_columns]
        normalized_x_data = scaler.fit_transform(x_data)
        imputer = KNNImputer(n_neighbors=5)
        imputed_normalized_array = imputer.fit_transform(normalized_x_data)
        imputed_array = scaler.inverse_transform(imputed_normalized_array)
        dataframe[self.__numeric_columns+self.__interest_columns] = imputed_array
        return dataframe

    
    def create_history_column(self, dataframe, no_of_seconds=5):
        for column in self.__interest_columns:
            new_column = "Average_" + column
            history_column = []
            base_column = dataframe[column]
            # This loop creates history values for the first k rows,
            # assuming that values before the recorded time were zero.
            for i in range(no_of_seconds):
                history_value = base_column.iloc[:i + 1].sum() / no_of_seconds
                history_column.append(history_value)

            ## USE INSTANTANEOUS FOR THROUGHPUT
            ## USE THIS FUNCTION FOR OTHER FEATURES
            ## CHANGE TO WEIGHTED AVERAGE
            for i in range(no_of_seconds + 1, len(dataframe) + 1):
                history_value = base_column.iloc[i - no_of_seconds:i].mean()
                history_column.append(history_value)
            dataframe[new_column] = history_column
        return dataframe

    def one_hot_encode(self):
        categorical_features = self.__categorical_columns+["Timestamp", "session"]
        cols = [i for i in self.__df.columns if i not in categorical_features]
        self.__df[cols] = self.__df[cols].astype(float)
        self.__df = pd.get_dummies(self.__df)
        self.__categorical_columns = list(set(self.__df.drop(columns=["Timestamp"])).difference(\
            self.__df[self.__numeric_columns+self.__geo_columns+self.__interest_columns]))

    def scale_data(self):
        scaler = StandardScaler()
        x_data = self.__df[self.__numeric_columns+self.__geo_columns+\
            self.__historic_columns+self.__interest_columns]
        normalized_x_data = scaler.fit_transform(x_data)
        self.__df[self.__numeric_columns+self.__geo_columns+self.__historic_columns+\
            self.__interest_columns] = normalized_x_data
        self.__scaler = scaler

    def create_grouped_sequences(self, history_length=10, horizon_length=5, 
     include_columns=[], exclude_columns=["Timestamp", "CellID", "DL_bitrate", "UL_bitrate"],
     y_columns=["Average_DL_bitrate", "Average_UL_bitrate"], inplace=False):
        groups = self.__df.groupby("session")
        for name, group in groups:
            x, y = self.create_sequences(dataframe=group, history_length=history_length,
             horizon_length=horizon_length, exclude_columns=exclude_columns,
              include_columns=include_columns, y_columns=y_columns, inplace=False)
            self.__x_sequences = self.__x_sequences + x
            self.__y_sequences = self.__y_sequences + y
        self.__x_sequences = np.array(self.__x_sequences)
        self.__y_sequences = np.array(self.__y_sequences)

    def create_sequences(self, dataframe=pd.DataFrame(), history_length=10, horizon_length=5, 
     include_columns=[], exclude_columns=[],
     y_columns=["Average_DL_bitrate", "Average_UL_bitrate"], inplace=False):

        if dataframe.empty:
            dataframe = self.__df
        if exclude_columns:
            include_columns = list(set(dataframe.columns).difference(exclude_columns))
        data = dataframe[include_columns]
        new_order = y_columns+list(set(include_columns).difference(y_columns))
        data = data.reindex(columns=new_order)
        x_sequences = []
        y_sequences = []
        for i in range(history_length, len(data)-horizon_length+1):
            x_sequences.append(data[i-history_length:i])
            # Not so sure about this ask for assistance.
            y_sequences.append(data[y_columns][i:i+horizon_length])
        if inplace:
            self.__x_sequences = np.array(x_sequences)
            self.__y_sequences = np.array(y_sequences)
            return None, None
        return x_sequences, y_sequences

    def get_df(self):
        return self.__df

    def get_scaler(self):
        return self.__scaler

    def get_x_sequences(self):
        return self.__x_sequences

    def get_y_sequences(self):
        return self.__y_sequences

    def save_scaler(self, filename="scaler.sav"):
        filepath = "src/saved.objects/"+filename
        pickle.dump(self.__scaler, open(filepath, "wb"))


if __name__ == "__main__":

    # Transform from raw data
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", index_col=None)
    pre_processor = WirelessDataPreProcessor(raw_data)
    imputed_data = pre_processor.get_df()
    pre_processor.save_scaler(filename="scaler_1.sav")
    imputed_data.to_csv("Datasets/Transformed/4G_data_ASN.csv", index=False, encoding="utf-8")

    # import transformed data and create sequences.
    # data = pd.read_csv("Datasets/Transformed/4G_data_ASN.csv", index_col=None, encoding="utf-8")
    # data_processor = WirelessDataPreProcessor(data, manual_mode=True)
    # test_df = data_processor.get_df()
    # test_df = test_df[test_df["session"]==0]
    # x, y = data_processor.create_sequences(dataframe=test_df,
    # exclude_columns=["Timestamp", "CellID", "DL_bitrate", "UL_bitrate"],
    #  y_columns=["Average_DL_bitrate", "Average_UL_bitrate"])
    # print(y)