import configparser
import sys
import tensorflow as tf
from time import time
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
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()])
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

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
        mse, mae = test[0], test[1]
        mape = self.get_mape(y, predicted_y)
        average_bias = self.get_average_bias(y, predicted_y)
        model_name = self._model_name+"_"+suffix
        self._results[0] = model_name
        self._results[4] = time_to_predict
        self._results[5] = mse
        self._results[6] = mae
        self._results[7] = average_bias
        self._results[8] = mape
        self.write_to_csv()
        self.save_output(predicted_y, model_name+"_predicted_y")
        self.save_output(self._test_y, model_name+"_true_y")


        