import tensorflow as tf
from abc import ABC, abstractclassmethod
import os


class ModelFramework(ABC):
    @abstractclassmethod
    def __init__(self):
        self._model = tf.keras.Sequential()
        self._model_name = "DEFAULT_NAME"
        self._preprocessor = None
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
        self._results = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
        print(self._model_name, "\n===============\n",self._model.metrics_names, "\n===============\n" ,self._results, "\n===============\n")

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

    def get_model_size(self):
        filepath = 'src/saved.objects/{}.hdf5'.format(self._model_name)
        file_stats = os.stat(filepath)
        model_size = file_stats.st_size/(1024*1024)
        return model_size

