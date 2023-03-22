import configparser
import sys
import tensorflow as tf
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.multistage_regression_model import MultiStageLSTM
from training_models.classifier import ThroughputClassifier
from training_models.baseline_regression_model import BaselineLSTM


class StandarizedBaseline(BaselineLSTM):
    def build_model(self):
        self._model.add(tf.keras.layers.LSTM(96, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.4))
        self._model.add(tf.keras.layers.LSTM(48,return_sequences=True))
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