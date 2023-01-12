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
    def __init__(self, raw_data=pd.DataFrame(), model_name="label_predictor_univariate"):
        super().__init__()
        self._raw_data = raw_data
        self._model_name = model_name

    def pre_process(self, preprocessor=None, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None):
        if preprocessor:
            self._preprocessor = preprocessor
        else:
            scaler_file_name = self._model_name + ".sav"
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
            use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)

        # Basic formatting
        self._train_x, self._train_y = self._preprocessor.get_label_predictor_train()
        self._test_x, self._test_y = self._preprocessor.get_label_predictor_test()
        self._train_x = np.array(self._train_x, ndmin=2)
        self._test_x = np.array(self._test_x, ndmin=2)
        print(self._test_x[0])
        self._train_y = np.array(self._train_y, ndmin=2)
        self._test_y = np.array(self._test_y, ndmin=2)
        print(self._train_y.shape)
        print(self._train_x.shape)

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
        self._model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self._model.summary()

    def train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._tensorboard = TensorBoard(log_dir="src/logs/{}".format(self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/{}.hdf5'.format(self._model_name), verbose = 1, save_best_only=True)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size,
         validation_split=validation_split, verbose=1, callbacks=[self._checkpointer, self._tensorboard])

    def get_preformance_metrics(self):
        pass

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = LabelPredictor(raw_data)
    example.pre_process()
    example.build_model()
    example.train()
    example.test()