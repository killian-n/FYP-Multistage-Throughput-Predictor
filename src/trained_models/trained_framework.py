import warnings
import tensorflow as tf
from abc import ABC, abstractclassmethod
import os
import numpy as np
import configparser
import csv
import pickle
config = configparser.ConfigParser()
config.read('project.env')

warnings.filterwarnings('ignore', category=UserWarning, append=True)

class TrainedFramework(ABC):
    @abstractclassmethod
    def __init__(self):
        self._model = tf.keras.Sequential()
        self._results = []
        self._model_name = "DEFAULT_NAME"
        self._model_size = 0
        self._output_shape = None
        self._test_x = None
        self._test_y = None
        self._scaler = None
    
    @abstractclassmethod
    def test(self):
        pass

    def __call__(self, inputs):
        inputs = self.scale(inputs)
        predictions = self._model(inputs)
        predictions = self.inverse_scale_predictions(predictions)
        return predictions
    
    def predict(self, x_sequences):
        x_sequences = self.scale(x_sequences)
        predictions = self._model.predict(x_sequences)
        predictions = self.inverse_scale_predictions(predictions)
        return predictions

    def set_model(self, model):
        self._model = model
        self.set_output_shape()
    
    def get_model(self):
        return self._model

    def get_test(self):
        return self._test_x, self._test_y
    
    def get_scaler(self):
        return
    
    def set_output_shape(self):
        for layer in self._model.layers:
            shape = layer.output_shape
        self._output_shape = shape

    def get_output_shape(self):
        return self._output_shape

    def set_test(self, test_x, test_y):
        self._test_x = test_x
        self._test_y = test_y

    def set_scaler(self, filepath=""):
        self._scaler = pickle.load(open(filepath, "rb"))
    
    def save_output(self,output,filename="DEFAULT_NAME_OUTPUTS"):
        filename = config["global"]["PROJECT_PATH"]+"Datasets/Final_Outputs/"+filename
        np.save(filename, output)

    def get_model_size(self):
        if not self._model_size:
            self.set_model_size()
        return self._model_size
    
    def set_model_size(self, filepath=""):
        saved_object_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_object_path[-1] not in ["\\", "/"]:
            saved_object_path += "/"
        if not filepath:
            filepath = '{}{}.hdf5'.format(saved_object_path, self._model_name)
        file_stats = os.stat(filepath)
        self._model_size = file_stats.st_size/(1024*1024)
    
    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)

    def inverse_scale_predictions(self, results):
        results_shape = results.shape
        results = np.squeeze(results).flatten()
        transform = np.zeros((len(results), self._scaler.n_features_in_))
        transform[:,0] = results
        results = self._scaler.inverse_transform(transform)[:,0]
        return results.reshape(results_shape)
    
    def scale(self, input_array, is_x=True):
        input_shape = input_array.shape
        if is_x:
            input_array = self._scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self._scaler.n_features_in_))
            transform[:,0] = input_array
            transform = self._scaler.transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

