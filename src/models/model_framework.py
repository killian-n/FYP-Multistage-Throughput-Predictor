import tensorflow as tf
from abc import ABC, abstractclassmethod


class ModelFramework(ABC):
    @abstractclassmethod
    def __init__(self):
        self._model = tf.keras.Sequential()
        self._model_name = "DEFAULT_NAME"
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
        for layer in self._model.layers:
            self._input_shape = layer.output_shape
            break
    
    def set_output_shape(self):
        for layer in self._model.layers:
            shape = layer.output_shape
        self._output_shape = shape

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        return self._output_shape

