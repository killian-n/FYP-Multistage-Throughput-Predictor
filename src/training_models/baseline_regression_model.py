import configparser
import sys
import tensorflow as tf
from time import time
from keras.utils.layer_utils import count_params
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
import pandas as pd
import pickle
config = configparser.ConfigParser()
config.read('project.env')
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
        epsilon = self.compute_epsilon()
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        # Option One
        # self._model.compile(optimizer="adam", loss=self.custom_loss(epsilon), metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        ####

        # Option Two
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        ####
        
        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        timer = TimingCallback()
        self._tensorboard = TensorBoard(log_dir="{}src/logs/{}".format(project_path, self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='{}src/saved.objects/{}.hdf5'.format(project_path, self._model_name), verbose = 1, save_best_only=False)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
         verbose=1, callbacks=[self._checkpointer, self._tensorboard, timer])
        self._train_time = sum(timer.logs)

    def __call__(self, inputs):
        inputs = self.scale(inputs)
        predictions = self._model(inputs)
        predictions = self.__inverse_scale_predictions(predictions)
        return predictions

    def predict(self, inputs, test=False):
        if not test:
            inputs = self.scale(inputs)
        predictions = self._model.predict(inputs)
        print("In {}'s predict, before scaling the prediction is", predictions[0])
        predictions = self.__inverse_scale_predictions(predictions)
        return predictions

    def set_train(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y
        self._train_x = self.scale(self._train_x)
        self._train_y = self.scale(self._train_y, is_x=False)

    def set_test(self, test_x, test_y):
        self._test_x = test_x
        self._test_y = test_y
        self._test_x = self.scale(self._test_x)
        self._test_y = self.scale(self._test_y, is_x=False)

    def test(self):
        trainable_params = count_params(self._model.trainable_weights)
        non_trainable_params = count_params(self._model.non_trainable_weights)
        predict_start = time()
        predicted_y = self._model.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
        # option One
        # mape, mse, mae = test[0], test[1], test[2]
        ######
        # Option Two
        mse, mae = test[0], test[1]
        mape = self.get_mape(self._test_y, predicted_y)
        ######

        average_bias = self.get_average_bias(self._test_y, predicted_y)
        model_size = self.get_model_size()
        self._results = [self._model_name, trainable_params, non_trainable_params, self._train_time, time_to_predict, mse, mae, average_bias, mape, model_size]
        self.write_to_csv()
        self.save_output(self.__inverse_scale_predictions(predicted_y), self._model_name+"_predicted_y")
        self.save_output(self.__inverse_scale_predictions(self._test_y), self._model_name+"_true_y")
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

        # Option One
        # mape, mse, mae = test[0], test[1], test[2]
        #######

        # Option Two
        mse, mae = test[0], test[1]
        mape = self.get_mape(y, predicted_y)
        ######

        average_bias = self.get_average_bias(y, predicted_y)
        model_name = self._model_name+"_"+suffix
        self._results[0] = model_name
        self._results[4] = time_to_predict
        self._results[5] = mse
        self._results[6] = mae
        self._results[7] = average_bias
        self._results[8] = mape
        self.write_to_csv()
        # self.save_output(self._train_x, self._model_name+"_train_x")
        # self.save_output(self._train_y, self._model_name+"_train_y")
        # self.save_output(self._test_x, self._model_name+"_test_x")
        self.save_output(self.__inverse_scale_predictions(predicted_y), model_name+"_predicted_y")
        self.save_output(self.__inverse_scale_predictions(y), model_name+"_true_y")
    
    def compute_epsilon(self, epsilon=50):
        scaler = self._preprocessor.get_scaler()
        transform = np.zeros((1, scaler.n_features_in_))
        transform[0,0] = epsilon
        transform = scaler.transform(transform)
        epsilon = transform[0,0]
        return epsilon
    
    def scale(self, input_array, is_x=True):
        input_shape = input_array.shape
        if not self._scaler:
            self._scaler = self._preprocessor.get_scaler()
        if is_x:
            input_array = self._scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self._scaler.n_features_in_))
            transform[:,0] = input_array
            transform = self._scaler.transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array
    
    def __inverse_scale_predictions(self, results):
        results_shape = results.shape
        scaler = self._preprocessor.get_scaler()
        results = np.squeeze(results).flatten()
        transform = np.zeros((len(results), scaler.n_features_in_))
        transform[:,0] = results
        results = scaler.inverse_transform(transform)[:,0]
        return results.reshape(results_shape)

    def get_mape(self, true, predicted, epsilon=50):
        epsilon = self.compute_epsilon(epsilon=epsilon)
        denominator = np.squeeze(true) + epsilon
        try:
            mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
        except Exception as e:
            print(e)
            mape = "n/a"
        return mape
    
    def get_performance_metrics(self):
        return self._results
    
    def set_scaler(self, filepath=""):
        self._scaler = pickle.load(open(filepath, "rb"))


if __name__=="__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="univariate_scaler.sav")
    example = BaselineLSTM(preprocessor=pre_processor, model_name="unop_univariate_baseline")
    example.pre_process()
    example.build_model()
    example.train(epochs=70)
    example.test()
