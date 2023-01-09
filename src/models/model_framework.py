import tensorflow as tf
from abc import ABC, abstractclassmethod


class ModelFramework(ABC):
    @abstractclassmethod
    def __init__(self, raw_data):
        self._raw_data = raw_data
        self._model = tf.keras.Sequential()
        self._checkpointer = None
        self._train_x = []
        self._train_y = []
        self._test_x = []
        self._test_y = []
        self._results

    @abstractclassmethod
    def build_model(self):
        pass
    
    @abstractclassmethod
    def pre_process(self):
         pass

    @abstractclassmethod
    def train(self):
        pass

    def test(self):
        self._results = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
    
    @abstractclassmethod
    def get_preformance_metrics(self):
        pass

