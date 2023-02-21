import configparser
import sys
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.layer_utils import count_params
import numpy as np
from time import time
import csv

config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.model_framework import ModelFramework
from data_transformation.preprocessor import DataPreProcessor
from helper_functions.timer import TimingCallback

class SimpleLSTM(ModelFramework):
    def __init__(self, raw_data=pd.DataFrame(),preprocessor=None, model_name="lstm_univariate"):
        super().__init__()
        self._raw_data = raw_data
        self._model_name = model_name
        self._preprocessor = preprocessor

    def pre_process(self, include_features=[], predict=["DL_bitrate"],
     use_predict=True, manual_mode=False, scaler=None, history=10, horizon=5):
        if not self._preprocessor:
            scaler_file_name = self._model_name + ".sav"
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
             use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name, history=history, horizon=horizon)
        self._train_x, self._train_y = self._preprocessor.get_train_sequences()
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()

    def build_model(self):
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32,return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss="mean_absolute_percentage_error", metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

    def train(self, epochs=20, batch_size=100, validation_split=0.2):
        timer = TimingCallback()
        self._tensorboard = TensorBoard(log_dir="src/logs/{}".format(self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/{}.hdf5'.format(self._model_name), verbose = 1, save_best_only=True)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
         verbose=1, callbacks=[self._checkpointer, self._tensorboard, timer])
        self._train_time = sum(timer.logs)

    def test(self):
        trainable_params = count_params(self._model.trainable_weights)
        non_trainable_params = count_params(self._model.non_trainable_weights)
        predict_start = time()
        predicted_y = self._model.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
        mape, mse, mae = test[0], test[1], test[2]
        average_bias = self.get_average_bias(self._test_y, predicted_y)
        model_size = self.get_model_size()
        self._results = [self._model_name, trainable_params, non_trainable_params, self._train_time, time_to_predict, mse, mae, average_bias, mape, model_size]
        self.write_to_csv()

    def get_average_bias(self, true, predicted):
        average_bias = np.mean(np.squeeze(true)-predicted)
        return average_bias

    def get_performance_metrics(self):
        return self._results

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)


if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    # [1179.720947265625, 0.6580678224563599, 0.5000829100608826]
    # pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="base_model_multivariate_scaler",
    #  include_features=["CQI", "RSRQ","State", "NRxRSRQ", "Longitude", "Latitude", "Speed"], history=10, horizon=5)
    # [689.5520629882812, 0.6595041155815125, 0.5026030540466309]
    pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="base_model_univariate_scaler")
    # [16499.181640625, 0.6543842554092407, 0.4942069947719574]
    # pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="base_model_multivariate_scaler_2",
    #  include_features=["RSRQ","State", "NRxRSRQ", "Longitude", "Latitude"], history=10, horizon=5)
    # [709.9197998046875, 0.6604500412940979, 0.5026734471321106]
    # pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="base_model_multivariate_scaler_5",
    #  include_features=["RSRQ","State", "NRxRSRQ"], history=10, horizon=5)

    # pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="base_model_multivariate_scaler_4",
    #  include_features=["RSRQ", "NRxRSRQ"], history=10, horizon=5)
    train_x, train_y = pre_processor.get_train_sequences()
    test_x, test_y = pre_processor.get_test_sequences()
    example = SimpleLSTM(model_name="LSTM_Multiivariate_5")
    example.pre_process(preprocessed=True, train=[train_x, train_y], test=[test_x, test_y])
    example.build_model()
    example.train()
    example.test()