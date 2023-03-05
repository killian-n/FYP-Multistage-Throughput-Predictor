import configparser
import sys
import tensorflow as tf
from time import time
from keras.callbacks import ModelCheckpoint, TensorBoard
import pandas as pd
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from training_models.model_framework import ModelFramework
from helper_functions.timer import TimingCallback
from data_transformation.preprocessor import DataPreProcessor

class BaselineLSTM(ModelFramework):
    def __init__(self, raw_data=pd.DataFrame(),preprocessor=None, model_name="lstm_univariate"):
        super().__init__()
        self._raw_data = raw_data
        self._model_name = model_name
        self._preprocessor = preprocessor
        self._scaler = None

    def pre_process(self, include_features=[], predict=["DL_bitrate"],
     use_predict=True, manual_mode=False, scaler=None, history=10, horizon=5):
        if not self._preprocessor:
            scaler_file_name = self._model_name + ".sav"
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
             use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name, history=history, horizon=horizon)
        self._train_x, self._train_y = self._preprocessor.get_train_sequences()
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()

    def build_model(self):
        epsilon = self.__compute_epsilon()
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        self._model.compile(optimizer="adam", loss=self.custom_loss(epsilon), metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

    def train(self, epochs=70, batch_size=100, validation_split=0.2):
        timer = TimingCallback()
        self._tensorboard = TensorBoard(log_dir="src/logs/{}".format(self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/{}.hdf5'.format(self._model_name), verbose = 1, save_best_only=False)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
         verbose=1, callbacks=[self._checkpointer, self._tensorboard, timer])
        self._train_time = sum(timer.logs)

    def test(self):
        super().test()
        low_test_x, low_test_y = self._preprocessor.get_low_test_sequences()
        med_test_x, med_test_y = self._preprocessor.get_medium_test_sequences()
        high_test_x, high_test_y = self._preprocessor.get_high_test_sequences()
        self.__class_tests(low_test_x, low_test_y, "low")
        self.__class_tests(med_test_x, med_test_y, "medium")
        self.__class_tests(high_test_x, high_test_y, "high")

    def __class_tests(self, x, y, suffix="low"):
        predict_start = time()
        predicted_y = self._model.predict(x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/y.shape[0]
        test = self._model.evaluate(x, y, batch_size=100)
        mape, mse, mae = test[0], test[1], test[2]
        average_bias = self.get_average_bias(y, predicted_y)
        model_name = self._model_name+"_"+suffix
        self._results[0] = model_name
        self._results[4] = time_to_predict
        self._results[5] = mse
        self._results[6] = mae
        self._results[7] = average_bias
        self._results[8] = mape
        self.write_to_csv()
        self.save_output(self._train_x, self._model_name+"_train_x")
        self.save_output(self._train_y, self._model_name+"_train_y")
        self.save_output(self._test_x, self._model_name+"_test_x")
        self.save_output(predicted_y, model_name+"_predicted_y")
        self.save_output(self._test_y, model_name+"_true_y")


if __name__=="__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="THE_SAVIOR.sav")
    example = BaselineLSTM(preprocessor=pre_processor, model_name="THE_SAVIOUR")
    example.pre_process()
    example.build_model()
    example.train(epochs=70)
    example.test()
