import tensorflow as tf
from abc import ABC, abstractclassmethod
import os
import numpy as np
from keras import backend
from time import time
from keras.utils.layer_utils import count_params
import configparser
import csv
config = configparser.ConfigParser()
config.read('.env')

class ModelFramework(ABC):
    @abstractclassmethod
    def __init__(self):
        self._model = tf.keras.Sequential()
        self._model_name = "DEFAULT_NAME"
        self._preprocessor = None
        self._is_scaled = False
        self._checkpointer = None
        self._tensorboard = None
        self._train_x = []
        self._train_y = []
        self._test_x = []
        self._test_y = []
        self._results = None
        self._input_shape = ()
        self._output_shape = ()
        self._train_time = 0

    @abstractclassmethod
    def build_model(self):
        pass
    
    @abstractclassmethod
    def pre_process(self):
         pass

    @abstractclassmethod
    def train(self):
        pass

    @abstractclassmethod
    def get_performance_metrics(self):
        pass

    def __call__(self, inputs):
        return self._model(inputs)

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
        self.save_output(self.inverse_scale_predictions(predicted_y), self._model_name+"_predicted_y")
        self.save_output(self.inverse_scale_predictions(self._test_y), self._model_name+"_true_y")

    def set_model(self, model):
        self._model = model
        self.set_input_shape()
        self.set_output_shape()

    def get_model(self):
        return self._model

    def set_input_shape(self):
        self._input_shape = self._train_x.shape
    
    def set_output_shape(self):
        for layer in self._model.layers:
            shape = layer.output_shape
        self._output_shape = shape

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape

    def get_train(self):
        return self._train_x, self._train_y

    def get_test(self):
        return self._test_x, self._test_y

    def set_train(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y

    def set_test(self, test_x, test_y):
        self._test_x = test_x
        self._test_y = test_y

    def get_preprocessor(self):
        return self._preprocessor
    
    def save_output(self,output,filename="DEFAULT_NAME_OUTPUTS"):
        filename = "Datasets/Final_Outputs/"+filename
        np.save(filename, output)

    def get_model_size(self):
        filepath = 'src/saved.objects/{}.hdf5'.format(self._model_name)
        file_stats = os.stat(filepath)
        model_size = file_stats.st_size/(1024*1024)
        return model_size
    
    def compute_epsilon(self, epsilon=50):
        scaler = self._preprocessor.get_scaler()
        transform = np.zeros((1, scaler.n_features_in_))
        transform[0,0] = epsilon
        transform = scaler.transform(transform)
        epsilon = transform[0,0]
        return epsilon
    
    def scale(self, input_array):
        input_shape = input_array.shape
        self._scaler = self._preprocessor.get_scaler()
        input_array = self._scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        return input_array
    
    def inverse_scale_predictions(self, results):
        results_shape = results.shape
        scaler = self._preprocessor.get_scaler()
        results = np.squeeze(results).flatten()
        transform = np.zeros((len(results), scaler.n_features_in_))
        transform[:,0] = results
        results = scaler.inverse_transform(transform)[:,0]
        return results.reshape(results_shape)
    
    def custom_loss(self, dl_epsilon):
        def mean_absolute_percentage_error(y_true, y_pred):
            """"""
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.cast(y_true, y_pred.dtype)
            diff = tf.abs(
                (y_true - y_pred) / backend.maximum(tf.abs(y_true+dl_epsilon), backend.epsilon())
            )
            return 100.0 * backend.mean(diff, axis=-1)
        return mean_absolute_percentage_error
    
    def get_average_bias(self, true, predicted):
        average_bias = np.mean(np.squeeze(true)-predicted)
        return average_bias

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)