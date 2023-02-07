import numpy as np
import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import random
pd.options.mode.chained_assignment = None


class DataPreProcessor:
    def __init__(self, dataframe, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="univarte_scaler.sav", random_seed=130, history=10, horizon=5, sparse=False):

        # Metadata
        metadata = ["Timestamp", "session", "movement_type"]
        self.__random_seed = random_seed
        self.__history_length = history
        self.__horizon_length = horizon
        self.__sparse = sparse
        
        # Working dataframes
        self.__df = dataframe[include_features+predict+metadata]
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
        self.__df = self.__df.replace({"-": np.nan, "": np.nan})
        try:
            self.__df["Timestamp"] = pd.to_datetime(self.__df["Timestamp"],
             format="%Y.%m.%d_%H.%M.%S")
        except Exception as e:
            pass
        # Potential memory issue with float64
        self.__df[self.__numeric_features+self.__geo_features] \
            = self.__df[self.__numeric_features+self.__geo_features].astype("float64")
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

        # Output variables for use in models
        self.__features = []

        self.__scaler = scaler
        self.__scaler_length = 0
        self.__scaler_file_name = scaler_file_name
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

        # Train
        self.__y_train_labels = []
        # Sequence, targets after balancing
        self.__x_train_balanced = []
        self.__y_train_balanced = []

        # Test
        self.__y_test_labels = []
        # Sequence, targets after balancing
        self.__x_test_balanced = []
        self.__y_test_balanced = []

        if not manual_mode:
            random.seed(self.__random_seed)
            self.train_test_split()
            self.do_all_preprocessing(self.__train, self.__test, sparse=sparse)

    def do_all_preprocessing(self, train, test, sparse=False):
            if self.__categorical_features:
                self.one_hot_encode()
                print("Categorical Features Included!!")
            self.__train = train
            self.__test = test
            self.__train = self.impute_and_normalise(dataframe=self.__train)
            self.__test = self.impute_and_normalise(dataframe=self.__test, test=True, scaler=self.__scaler)
            self.__train = self.create_averaged_features(dataframe=self.__train)
            self.__test = self.create_averaged_features(dataframe=self.__test)
            self.__x_train, self.__y_train = self.create_sequences(self.__train, self.__history_length, self.__horizon_length)
            self.__x_test, self.__y_test = self.create_sequences(self.__test, self.__history_length, self.__horizon_length)
            self.__y_train_labels = self.create_labels(self.__y_train, sparse=sparse)
            self.__y_test_labels = self.create_labels(self.__y_test, sparse=sparse)
            self.__x_train_balanced, self.__y_train_balanced = self.balance_labels(self.__x_train, self.__y_train_labels, sparse=sparse, ignore_min_size=False)
            self.__x_test_balanced, self.__y_test_balanced = self.balance_labels(self.__x_test, self.__y_test_labels, train=False, ignore_min_size=False, sparse=sparse)
            self.separate_by_label(train=True, sparse=sparse)
            self.separate_by_label(train=False, sparse=sparse)
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
        self.__df[cols] = self.__df[cols].astype("float64")
        self.__df = pd.get_dummies(self.__df)
        self.__categorical_features = list(set(self.__df.drop(columns=["Timestamp"])).difference(\
            self.__df[self.__numeric_columns+self.__geo_columns]))
    
    def impute_and_normalise(self, dataframe, test=False, scaler=None, return_true_values=False):
        # IMPORTANT: See Data_Expolration notebook for reasoning on imputation choices

        # Is a scaler provided? If no and this is test data then throw an error.
        if not scaler:
            if test:
                print("Error, test data must use a prior scaler created on the training data.")
                return None
            # scaler = MinMaxScaler((-1,1))
            scaler = RobustScaler()
        
        if "CQI" in self.__numeric_features:
            dataframe["CQI"].fillna(dataframe["CQI"].min(), inplace=True)
        
        if "SNR" in self.__numeric_features:
            dataframe["SNR"].fillna(dataframe["SNR"].min(), inplace=True)

        if "RSSI" in self.__numeric_features:
            dataframe["RSSI"].fillna(dataframe["RSSI"].min(), inplace=True)

        # Checking to see if the y variable is being used to predict itself.
        if self.__use_predict:
            order = self.__predict+self.__numeric_features[:-(len(self.__predict))]+self.__geo_features+self.__categorical_features
        else:
            order = self.__numeric_features+self.__geo_features+self.__categorical_features
        x_data = dataframe[order]
        new_values = scaler.fit_transform(x_data)

        # NRxRSRP and NRxRSRQ require KNN-Imputation.
        require_KNN = ["NRxRSRP", "NRxRSRQ", "RSRP", "RSRQ"]
        require_KNN = [i for i in require_KNN if i in self.__numeric_features]
        if require_KNN:
            imputer = KNNImputer(n_neighbors=5)
            new_values = imputer.fit_transform(new_values)

        # Require the original values
        if return_true_values:
            new_values = scaler.inverse_transform(new_values)

        # Reasssigning scaled and imputed data to the dataframe
        dataframe[order] = new_values
        dataframe[order].astype("float64")
        if not test:
            self.__scaler = scaler
            self.__scaler_length = len(x_data.columns)
        return dataframe

    # MORE THOUGHT NEEDED ON USAGE
    def scale(self, dataframe, test=False, scaler=None):
        if not scaler:
            if test:
                print("Error, test data must use a prior scaler created on the training data.")
                return None
            scaler = MinMaxScaler()
        if self.__use_predict:
            x_data = dataframe[self.__predict+self.__numeric_features[:-(len(self.__predict))]+self.__geo_features+self.__categorical_features]
        else:
            x_data = dataframe[self.__numeric_features+self.__geo_features+self.__categorical_features]
        scaler_length = len(x_data.columns)
        normalised_x_data = scaler.fit_transform(x_data)
        dataframe[self.__numeric_features] = normalised_x_data
        dataframe[self.__numeric_features].astype("float64")
        if not test:
            self.__scaler = scaler
            self.__scaler_length = scaler_length
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
    
    def create_sequences(self, dataframe, history_length, horizon_length):
        if self.__history_length == 0 and self.__horizon_length == 0:
            if history_length != self.__history_length or horizon_length != self.__horizon_length:
                self.__history_length = history_length
                self.__horizon_length = horizon_length
        # x, y are lists of dataframes
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
            numeric_features = self.__numeric_features[:-(len(self.__predict))]
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
            # FOR NOW ASSUMING DL_bitrate IS THE ONLY FACTOR USED TO LABEL FUTURE THROUGHPUT
            # ALSO ASSUMES THAT DL_bitrate IS ALWAYS FIRST ITEM IN self.__predict
            average_throughput = (sum(transform[0])/len(transform[0]))/1000
            if average_throughput < 1:
                y_labels.append(label_dict["low"])
            elif average_throughput > 5:
                y_labels.append(label_dict["high"])
            else:
                y_labels.append(label_dict["medium"])
        return y_labels

    def balance_labels(self, x_sequences, labels, train=True, ignore_min_size=True, sparse=False):
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
        minimum = min(len(low_y), len(medium_y), len(high_y))
        
        if not ignore_min_size:
            if train:
                if minimum < 18000:
                    print("Minimum no. of examples of a label is", minimum)
                    print("training datasets are too small, rerunning preprocessing for better mix of data.")
                    self.train_test_split()
                    self.do_all_preprocessing(self.__train, self.__test, sparse=self.__sparse)
                    # Above function will continue to be run until values below meet the criteria.
                    return self.__x_train_balanced, self.__y_train_balanced
            # Test set is too small.
            else:
                if minimum < 4000:
                    print("Minimum no. of examples of a label is", minimum)
                    print("test datasets too small, rerunning preprocessing for better mix of data.")
                    self.train_test_split()
                    self.do_all_preprocessing(self.__train, self.__test, sparse=self.__sparse)
                    return self.__x_test_balanced, self.__y_test_balanced
        low_x = low_x[:minimum]
        low_y = low_y[:minimum]
        medium_x = medium_x[:minimum]
        medium_y = medium_y[:minimum]
        high_x = high_x[:minimum]
        high_y = high_y[:minimum]
        x = low_x + medium_x + high_x
        y = low_y + medium_y + high_y
        return x, y

    def separate_by_label(self, train=True, sparse=False):
        low_x = []
        medium_x = []
        high_x = []
        low_y = []
        medium_y = []
        high_y = []
        reverse_label_dict = {}
        if sparse:
            label_dict = self.__sparse_label_dict
        else:
            label_dict = self.__label_dict


        for key in label_dict:
            val = label_dict[key]
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

    def get_label_predictor_train(self):
        return np.array(self.__x_train_balanced), np.array(self.__y_train_balanced, ndmin=2)

    def get_label_predictor_test(self):
        return np.array(self.__x_test_balanced), np.array(self.__y_test_balanced, ndmin=2)

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

    def get_df(self):
        return self.__df

    def set_train(self, train_df=pd.DataFrame()):
        self.__train = train_df

    def set_test(self, test_df=pd.DataFrame()):
        self.__test = test_df

    def save_scaler(self, filename=None):
        if not filename:
            filename = self.__scaler_file_name
        filepath = "src/saved.objects/"+filename
        pickle.dump(self.__scaler, open(filepath, "wb"))

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", index_col=None)
    potential_seeds = [42, 119, 130, 155, 173]
    # check to 150
    # 130 best
    for i in [130]:
        print("\n\nrandom seed is", i)
        pre_processor = DataPreProcessor(raw_data, manual_mode=True, include_features=["SNR"], predict=["DL_bitrate"], random_seed=i)
        pre_processor.train_test_split()
        train = pre_processor.get_train()
        test = pre_processor.get_test()
        train = pre_processor.impute_and_normalise(train, return_true_values=True)
        # test = pre_processor.impute_and_normalise(test, scaler=pre_processor.get_scaler(), test=True, return_true_values=True)
        x_train_sequences, y_train_sequences = pre_processor.create_sequences(train, 10, 5)
        x_train_sequences = np.array(x_train_sequences)
        print(x_train_sequences[0])
        print("=======")
        k = x_train_sequences[:,:,0][0]
        print(k)
        with open("Debug3.txt", "w") as f:
            f.write(np.array2string(k, separator=","))
            print("X Size", x_train_sequences.shape)
            # print(x_train_sequences[:10])
        # 42 119 130
