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
    def get_preformance_metrics(self):
        pass

    def __call__(self, inputs):
        return self._model(inputs)

    def test(self):
        self._results = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
        print(self._model_name, "\n===============\n",self._model.metrics_names, "\n===============\n" ,self._results, "\n===============\n")

    def get_model(self):
        return self._model
