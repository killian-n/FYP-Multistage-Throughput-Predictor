import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.model_framework import ModelFramework
from data_transformation.preprocessor import DataPreProcessor

class SimpleLSTM(ModelFramework):
    def __init__(self, raw_data):
        super().__init__(raw_data)

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="univarte_scaler.sav"):
        self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
         use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)
        self._train_x, self._train_y = self._preprocessor.get_train_sequences()
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()
        self._train_x = np.array(self._train_x)
        self._train_y = np.array(self._train_y)
        self._test_x = np.array(self._test_x)
        self._test_y = np.array(self._test_y)

    def build_model(self):
        self._model.add(tf.keras.layers.LSTM(64, activation="relu", input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.LSTM(32, activation="relu",return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1]))
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()

    def train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/SimpleLSTM_univariate.hdf5', verbose = 1, save_best_only=True)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1, callbacks=[self._checkpointer])

    def get_preformance_metrics(self):
        pass


if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = SimpleLSTM(raw_data)
    example.pre_process()
    example.build_model()
    example.train()
    example.test()
