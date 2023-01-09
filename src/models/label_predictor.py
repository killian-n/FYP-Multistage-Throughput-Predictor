import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.model_framework import ModelFramework
from data_transformation.preprocessor import DataPreProcessor

class LabelPredictor(ModelFramework):
    def __init__(self, raw_data):
        super().__init__(raw_data)
        self._train_labels = None
        self._test_labels = None

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="label_predictor_scaler.sav"):
        self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
         use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)
        self._train_x, self._train_labels = self._preprocessor.get_train_labels()
        self._test_x, self._test_labels = self._preprocessor.get_test_labels()
        self._train_x = np.array(self._train_x)
        self._test_x = np.array(self._test_x)
        self._train_labels = np.array(self._train_labels, ndmin=2).T
        self._test_labels = np.array(self._test_labels, ndmin=2).T

        
    def build_model(self):
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64, input_shape=(self._train_x.shape[1:]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32, input_shape=(self._train_x.shape[1:]),return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.BatchNormalization())
        self._model.add(tf.keras.layers.Dense(32))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(3, activation="softmax"))
        self._model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self._model.summary()

    def train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/LabelPredictor_univariate.hdf5', verbose = 1, save_best_only=True)
        self._model.fit(self._train_x, self._train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1, callbacks=[self._checkpointer])

    def test(self, batch_size=100):
        self._results = self._model.evaluate(self._test_x, self._test_labels, batch_size=batch_size)

    def get_preformance_metrics(self):
        pass


if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = LabelPredictor(raw_data)
    example.pre_process()
    example.build_model()
    example.train()
    example.test()