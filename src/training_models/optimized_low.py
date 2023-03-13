import configparser
import sys
import tensorflow as tf
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.multistage_regression_model import MultiStageLSTM


class optimizedLowRegressionModel(MultiStageLSTM):
    def build_model(self):
        epsilon = self.__compute_epsilon()
        self._model.add(tf.keras.layers.LSTM(128, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.2))
        self._model.add(tf.keras.layers.LSTM(64,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.2))
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(32, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dropout(.1))
        self._model.add(tf.keras.layers.Dense(24, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dense(16, activation="sigmoid"))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()