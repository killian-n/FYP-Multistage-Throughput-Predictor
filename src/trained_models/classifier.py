import configparser
import sys
from time import time
import csv
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.trained_framework import TrainedFramework


class TrainedClassifier(TrainedFramework):
    def __init__(self, model_name="univariate_classifier"):
        super().__init__()
        self._model_name = model_name

    def __call__(self, inputs):
        inputs = self.scale(inputs)
        predictions = self._model(inputs)
        return predictions
    
    def predict(self, x_sequences):
        x_sequences = self.scale(x_sequences)
        predictions = self._model.predict(x_sequences)
        return predictions

    def test(self):
        predict_start = time()
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self.scale(self._test_x), self._test_y, batch_size=100)
        accuracy = test[1]
        model_size = self.get_model_size()
        print(self._model_name)
        self._results = [self._model_name, time_to_predict, "n/a", "n/a", "n/a","n/a", model_size, accuracy]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")

    def set_test(self, test_x, test_y, loss="sparse_categorical_crossentropy"):
        self._test_x = test_x
        self._test_y = test_y
        if loss == "sparse_categorical_crossentropy":
            self._test_y = self._test_y.T

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)