import tensorflow as tf
from abc import ABC, abstractclassmethod
import os
import numpy as np
from keras import backend
import configparser
import csv
import pickle
config = configparser.ConfigParser()
config.read('project.env')

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
    
    @abstractclassmethod
    def test(self):
        pass

    def __call__(self, inputs):
        return self._model(inputs)

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
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        filename = "{}Datasets/Final_Outputs/".format(project_path)+filename
        np.save(filename, output)

    def get_model_size(self):
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        filepath = '{}src/saved.objects/{}.hdf5'.format(project_path, self._model_name)
        file_stats = os.stat(filepath)
        model_size = file_stats.st_size/(1024*1024)
        return model_size
    
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

    def set_scaler(self, filepath=""):
        self._scaler = pickle.load(open(filepath, "rb"))