import configparser
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.multistage_regression_model import MultiStageLSTM
from training_models.classifier import ThroughputClassifier
from training_models.baseline_regression_model import BaselineLSTM
from data_transformation.preprocessor import DataPreProcessor

class optimizedBaseline(BaselineLSTM):
    def build_model(self):
        #Univariate
        if self._train_x.shape[2]==1:
            self._model.add(tf.keras.layers.LSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.2))
            self._model.add(tf.keras.layers.LSTM(96,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.4))
            self._model.add(tf.keras.layers.LSTM(112,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.1))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(16, activation="relu"))
            self._model.add(tf.keras.layers.Dense(16, activation="relu"))
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        #Multivariate
        else:
            self._model.add(tf.keras.layers.LSTM(128, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.4))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.3))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.2))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.2))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(16, activation="relu"))
            self._model.add(tf.keras.layers.Dropout(0))
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

class optimizedClassifierModel(ThroughputClassifier):
    def build_model(self, loss="categorical_crossentropy"):
        self._model.add(tf.keras.layers.LSTM(160, input_shape=(self._train_x.shape[1], 1), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.2))
        self._model.add(tf.keras.layers.LSTM(128,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.0))
        self._model.add(tf.keras.layers.LSTM(80,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.3))
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(48, activation="relu"))
        self._model.add(tf.keras.layers.Dropout(0.0))
        self._model.add(tf.keras.layers.Dense(56, activation="relu"))
        self._model.add(tf.keras.layers.Dropout(0.3))
        self._model.add(tf.keras.layers.Dense(48, activation="relu"))
        self._model.add(tf.keras.layers.Dropout(0.4))
        self._model.add(tf.keras.layers.Dense(3, name="OUT_{}".format(self._model_name), activation="softmax"))
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss, metrics=["accuracy"])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()


class optimizedLowRegressionModel(MultiStageLSTM):
    def build_model(self):
        #Univariate
        if self._train_x.shape[2]==1:
            self._model.add(tf.keras.layers.LSTM(240, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.0))
            self._model.add(tf.keras.layers.LSTM(128,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.1))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.4))
            self._model.add(tf.keras.layers.LSTM(80,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.0))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        #Multivariate
        else:
            self._model.add(tf.keras.layers.LSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.2))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.1))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.1))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.1))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

class optimizedMediumRegressionModel(MultiStageLSTM):
    def build_model(self):
        #Univariate
        if self._train_x.shape[2]==1:
            self._model.add(tf.keras.layers.LSTM(240, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.3))
            self._model.add(tf.keras.layers.LSTM(128,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.1))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.4))
            self._model.add(tf.keras.layers.LSTM(80,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.0))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        #Multivariate
        else:
            self._model.add(tf.keras.layers.LSTM(208, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.3))
            self._model.add(tf.keras.layers.LSTM(96,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.0))
            self._model.add(tf.keras.layers.LSTM(80,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.3))
            self._model.add(tf.keras.layers.LSTM(112,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.4))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

class optimizedHighRegressionModel(MultiStageLSTM):
    def build_model(self):
        #Univariate
        if self._train_x.shape[2]==1:
            self._model.add(tf.keras.layers.LSTM(192, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.3))
            self._model.add(tf.keras.layers.LSTM(112,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.3))
            self._model.add(tf.keras.layers.LSTM(80,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(0.1))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        #Multivariate
        else:
            self._model.add(tf.keras.layers.LSTM(240, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.2))
            self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
            self._model.add(tf.keras.layers.Dropout(.2))
            self._model.add(tf.keras.layers.Flatten())
            self._model.add(tf.keras.layers.Dense(48, activation="relu"))
            self._model.add(tf.keras.layers.Dropout(0.2))
            self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
            self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

# if __name__ == "__main__":
#     raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
#     all_network_preprocessor = DataPreProcessor(raw_data, scaler_file_name="all_network_features.sav", include_features=["NRxRSRQ", "RSRQ","RSRP" ,"SNR", "CQI", "RSSI", "NRxRSRP"], name="all_network_unscaled")
#     example = optimizedClassifierModel(model_name="solo_all_network_classifier", preprocessor=all_network_preprocessor,sparse=True)
#     train_x = np.load("Datasets/Training/all_network_classifier_train_x.npy")
#     train_y = np.load("Datasets/Training/all_network_classifier_train_y.npy")
#     test_x = np.load("Datasets/Testing/all_network_classifier_test_x.npy")
#     test_y = np.load("Datasets/Testing/all_network_classifier_test_y.npy")
#     example.set_train(train_x, train_y)
#     example.set_test(test_x, test_y)
#     example.build_model(loss="sparse_categorical_crossentropy")
#     example.train()
#     example.test()