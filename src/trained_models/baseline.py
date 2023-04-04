import configparser
import sys
from time import time
import numpy as np
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from trained_models.trained_framework import TrainedFramework

class TrainedBaseline(TrainedFramework):
    def __init__(self, model_name="baseline"):
        super().__init__()
        self._model_name = model_name
        self._low_test_x = None
        self._low_test_y = None
        self._medium_test_x = None
        self._medium_test_y = None
        self._high_test_x = None
        self._high_test_y = None

    def test(self):
        predict_start = time()
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self.scale(self._test_x), self.scale(self._test_y,is_x=False), batch_size=100)
        mse, mae = test[0], test[1]
        mape = self.get_mape(self._test_y, predicted_y)
        mean_residuals = np.mean(self.get_residuals(self._test_y, predicted_y))
        model_size = self.get_model_size()
        self._results = [self._model_name, time_to_predict, mse, mae, mean_residuals, mape, model_size]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")
        self.__class_tests(self._low_test_x, self._low_test_y, "low")
        self.__class_tests(self._medium_test_x, self._medium_test_y, "medium")
        self.__class_tests(self._high_test_x, self._high_test_y, "high")

    def __class_tests(self, x, y, suffix="low"):
        predict_start = time()
        predicted_y = self.predict(x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/y.shape[0]
        test = self._model.evaluate(self.scale(x), self.scale(y, is_x=False), batch_size=100)
        mse, mae = test[0], test[1]
        mape = self.get_mape(y, predicted_y)
        mean_residuals = np.mean(self.get_residuals(y, predicted_y))
        model_name = self._model_name+"_"+suffix
        self._results[0] = model_name
        self._results[1] = time_to_predict
        self._results[2] = mse
        self._results[3] = mae
        self._results[4] = mean_residuals
        self._results[5] = mape
        self.write_to_csv()
        self.save_output(predicted_y, model_name+"_predicted_y")
        self.save_output(y, model_name+"_true_y")
    
    def compute_epsilon(self, epsilon=50):
        transform = np.zeros((1, self._scaler.n_features_in_))
        transform[0,0] = epsilon
        transform = self._scaler.transform(transform)
        epsilon = transform[0,0]
        return epsilon
    
    def get_mape(self, true, predicted, epsilon=50):
        epsilon2 = self.compute_epsilon(epsilon=epsilon)
        print("EPSILON", epsilon2)
        denominator = np.squeeze(true) + epsilon
        try:
            mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
        except Exception as e:
            print(e)
            mape = "n/a"
        return mape
    
    def get_residuals(self, true, predicted):
        return np.squeeze(true)-predicted
    
    def get_performance_metrics(self):
        return self._results
    
    def set_low_test(self, x, y):
        self._low_test_x = x 
        self._low_test_y = y

    def set_medium_test(self, x, y):
        self._medium_test_x = x 
        self._medium_test_y = y

    def set_high_test(self, x, y):
        self._high_test_x = x 
        self._high_test_y = y