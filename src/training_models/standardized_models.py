import configparser
import sys
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.multistage_regression_model import MultiStageLSTM
from data_transformation.preprocessor import DataPreProcessor
from training_models.classifier import ThroughputClassifier
from training_models.baseline_regression_model import BaselineLSTM


class StandarizedBaseline(BaselineLSTM):
    def build_model(self):
        self._model.add(tf.keras.layers.LSTM(96, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.4))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.4))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.2))
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(24, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dense(24, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dropout(.1))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()


class standardizedMultistageRegression(MultiStageLSTM):
    def build_model(self):
        self._model.add(tf.keras.layers.LSTM(96, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.4))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.4))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(0.2))
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(24, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dense(24, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dropout(.1))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()


if __name__ == "__main__":
    # HUGE NOTE:: Training multiple models in the same scripts is messing with model performance. Execute the script with only 1 model loded at a time for now.
    scaler = pickle.load(open("src/saved.objects/all_network_features.sav", "rb"))
    def inverse_scale(input_array, is_x=True, scaler=None):
        input_shape = input_array.shape
        if is_x:
                input_array = scaler.inverse_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), scaler.n_features_in_))
            transform[:,0] = input_array
            transform = scaler.inverse_transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

    # LOW
    # example = standardizedMultistageRegression(model_name="solo_low_all_network")
    # train_x = np.load("Datasets/Training/all_network_low_train_x.npy")
    # train_y = np.load("Datasets/Training/all_network_low_train_y.npy")
    # test_x = np.load("Datasets/Testing/all_network_low_test_x.npy")
    # test_y = np.load("Datasets/Testing/all_network_low_test_y.npy")
    # train_x = inverse_scale(train_x, scaler=scaler)
    # train_y = inverse_scale(train_y, is_x=False, scaler=scaler)
    # test_x = inverse_scale(test_x, scaler=scaler)
    # test_y = inverse_scale(test_y, is_x=False, scaler=scaler)
    # example.set_train(train_x, train_y)
    # example.set_test(test_x, test_y)
    # example.build_model()
    # example.train()
    # example.test()

    # Medium
    # example = standardizedMultistageRegression(model_name="solo_medium_all_network")
    # train_x = np.load("Datasets/Training/all_network_medium_train_x.npy")
    # train_y = np.load("Datasets/Training/all_network_medium_train_y.npy")
    # test_x = np.load("Datasets/Testing/all_network_medium_test_x.npy")
    # test_y = np.load("Datasets/Testing/all_network_medium_test_y.npy")
    # train_x = inverse_scale(train_x, scaler=scaler)
    # train_y = inverse_scale(train_y, is_x=False, scaler=scaler)
    # test_x = inverse_scale(test_x, scaler=scaler)
    # test_y = inverse_scale(test_y, is_x=False, scaler=scaler)
    # example.set_train(train_x, train_y)
    # example.set_test(test_x, test_y)
    # example.build_model()
    # example.train()
    # example.test()

    # high
    # example = standardizedMultistageRegression(model_name="solo_high_all_network")
    # train_x = np.load("Datasets/Training/all_network_high_train_x.npy")
    # train_y = np.load("Datasets/Training/all_network_high_train_y.npy")
    # test_x = np.load("Datasets/Testing/all_network_high_test_x.npy")
    # test_y = np.load("Datasets/Testing/all_network_high_test_y.npy")
    # train_x = inverse_scale(train_x, scaler=scaler)
    # train_y = inverse_scale(train_y, is_x=False, scaler=scaler)
    # test_x = inverse_scale(test_x, scaler=scaler)
    # test_y = inverse_scale(test_y, is_x=False, scaler=scaler)
    # example.set_train(train_x, train_y)
    # example.set_test(test_x, test_y)
    # example.build_model()
    # example.train()
    # example.test()

    # Baseline
    example = StandarizedBaseline(model_name="solo_all_network")
    train_x = np.load("Datasets/Training/all_network_train_x.npy")
    train_y = np.load("Datasets/Training/all_network_train_y.npy")
    test_x = np.load("Datasets/Testing/all_network_test_x.npy")
    test_y = np.load("Datasets/Testing/all_network_test_y.npy")
    example.set_train(train_x, train_y)
    example.set_test(test_x, test_y)
    example.build_model()
    example.train()
    example.test()