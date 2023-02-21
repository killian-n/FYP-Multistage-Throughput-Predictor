import configparser
import sys
import tensorflow as tf
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from models.simple_LSTM import SimpleLSTM


class ComplexLSTM(SimpleLSTM):
    def build_model(self):
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss="mean_absolute_percentage_error", metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()
