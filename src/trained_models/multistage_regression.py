import configparser
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from time import time

config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.trained_framework import TrainedFramework

class TrainedRegression(TrainedFramework):
    def __init__(self, model_name="univariate_low"):
        super().__init__()
        self._model_name = model_name
    
    def test(self):
        predict_start = time()
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self.scale(self._test_x), self.scale(self._test_y, is_x=False), batch_size=100)
        mse, mae = test[0], test[1]
        mape = self.get_mape(self._test_y, predicted_y)
        mean_residuals = np.mean(self.get_residuals(self._test_y, predicted_y))
        model_size = self.get_model_size()
        self._results = [self._model_name, time_to_predict, mse, mae,mean_residuals, mape, model_size]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")

    def get_mape(self, true, predicted, epsilon=50):
        denominator = np.squeeze(true) + epsilon
        try:
            mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
        except Exception as e:
            print(e)
            mape = "n/a"
        return mape
    
    def get_residuals(self, true, predicted):
        return np.squeeze(true)-predicted


