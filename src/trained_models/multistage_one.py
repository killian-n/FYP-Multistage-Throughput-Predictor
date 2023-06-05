import configparser
import warnings
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from time import time
import csv
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.classifier import TrainedClassifier
from trained_models.multistage_regression import TrainedRegression
warnings.filterwarnings('ignore', category=UserWarning, append=True)

class MultistageOne:
    def __init__(self, model_name="multiOne"):
        self._model_name = model_name
        self._model_size = 0
        self._low_model_size = 0
        self._medium_model_size = 0
        self._high_model_size = 0
        self._classifier_size = 0
        self._test_x = None
        self._test_y = None
        self._classifier = TrainedClassifier(model_name=self._model_name+"_classifier")
        self._low_tp_model = TrainedRegression(model_name=self._model_name+"_low")
        self._medium_tp_model = TrainedRegression(model_name=self._model_name+"_medium")
        self._high_tp_model = TrainedRegression(model_name=self._model_name+"_high")
        self._output_shape = ()
        self._results = []
    
    def __call__(self, x_sequences):
        prob_tables = self._classifier(x_sequences).numpy()
        label = np.zeros_like(prob_tables)
        label[np.arange(len(prob_tables)), prob_tables.argmax(axis=1)] = 1
        
        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        low_result = self._low_tp_model(x_sequences)*low_labels
        medium_result = self._medium_tp_model(x_sequences)*med_labels
        high_result = self._high_tp_model(x_sequences)*high_labels
        result = low_result + medium_result + high_result
        return result

    def predict(self, x_sequences, no_of_batches=70):
        if no_of_batches > x_sequences.shape[0]:
            no_of_batches = 1
        predictions = np.zeros((x_sequences.shape[0], self._output_shape[1]))
        index = 0
        for arr in np.array_split(x_sequences, no_of_batches):
            result = self.__call__(arr)
            predictions[index:result.shape[0]+index, :] = result
            index += result.shape[0]
        return predictions

    def get_mse(self, true, predicted):
        mse = np.power(np.squeeze(true)-predicted, 2).mean()
        return mse

    def get_mae(self, true, predicted):
        mae = np.mean(np.abs(np.squeeze(true)-predicted))
        return mae

    def get_mape(self, true, predicted, epsilon=50):
        denominator = np.squeeze(true) + epsilon
        try:
            mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
        except Exception as e:
            print(e)
            mape = "n/a"
        return mape

    def test(self):
        model_size = 0
        self._output_shape = self._high_tp_model.get_output_shape()
        # Individual Tests
        self._classifier.test()
        self._low_tp_model.test()
        self._medium_tp_model.test()
        self._high_tp_model.test()
        
        model_size = self._classifier_size
        model_size += self._low_model_size
        model_size += self._medium_model_size
        model_size += self._high_model_size

        predict_start = time()
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        
        mse = self.get_mse(self._test_y, predicted_y)
        mae = self.get_mae(self._test_y, predicted_y)
        mean_residuals = np.mean(self.get_residuals(self._test_y, predicted_y))
        mape = self.get_mape(self._test_y, predicted_y)
        self._results = [self._model_name, time_to_predict, mse, mae, mean_residuals, mape, model_size]
        self.write_to_csv()
        self.write_datasets_to_csv(predicted_y)
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")

        low_test_x, low_test_y = self._low_tp_model.get_test()
        low_predicted = self.predict(low_test_x)
        self.save_output(low_predicted, self._model_name+"_ms_low_predicted_y")
        self.write_datasets_to_csv(low_predicted, low_test_x, low_test_y, self._model_name+"_ms_low")
        
        medium_test_x, medium_test_y = self._medium_tp_model.get_test()
        medium_predicted = self.predict(medium_test_x)
        self.save_output(medium_predicted, self._model_name+"_ms_medium_predicted_y")
        self.write_datasets_to_csv(medium_predicted, medium_test_x, medium_test_y, self._model_name+"_ms_medium")

        high_test_x, high_test_y = self._high_tp_model.get_test()
        high_predicted = self.predict(high_test_x)
        self.save_output(high_predicted, self._model_name+"_ms_high_predicted_y")
        self.write_datasets_to_csv(high_predicted, high_test_x, high_test_y, self._model_name+"_ms_high")
        
    def get_performance_metrics(self):
        return self._results
    
    def inverse_scale(self, input_array, is_x=True):
        input_shape = input_array.shape
        if is_x:
             input_array = self._scaler.inverse_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self._scaler.n_features_in_))
            transform[:,0] = input_array
            transform = self._scaler.inverse_transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)

    def save_output(self,output,filename="DEFAULT_NAME_OUTPUTS"):
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        filename = project_path +"Datasets/Final_Outputs/"+filename
        np.save(filename, output)

    def get_residuals(self, true, predicted):
        return np.squeeze(true)-predicted
    
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
    
    def set_classifier(self, filepath=""):
        model = tf.keras.models.load_model(filepath)
        self._classifier.set_model_size(filepath)
        self._classifier_size = self._classifier.get_model_size()
        self._classifier.set_model(model)
    
    def set_low_model(self, filepath=""):
        model = tf.keras.models.load_model(filepath)
        self._low_tp_model.set_model_size(filepath)
        self._low_model_size = self._low_tp_model.get_model_size()
        self._low_tp_model.set_model(model)
    
    def set_medium_model(self, filepath=""):
        model = tf.keras.models.load_model(filepath)
        self._medium_tp_model.set_model_size(filepath)
        self._medium_model_size = self._medium_tp_model.get_model_size()
        self._medium_tp_model.set_model(model)

    def set_high_model(self, filepath=""):
        model = tf.keras.models.load_model(filepath)
        self._high_tp_model.set_model_size(filepath)
        self._high_model_size = self._high_tp_model.get_model_size()
        self._high_tp_model.set_model(model)

    def set_low_test(self, x, y):
        self._low_tp_model.set_test(x, y)

    def set_medium_test(self, x, y):
        self._medium_tp_model.set_test(x, y)

    def set_high_test(self, x, y):
        self._high_tp_model.set_test(x, y)
    
    def set_classifier_test(self, x, y):
        self._classifier.set_test(x, y)

    def set_low_scaler(self, filepath=""):
        self._low_tp_model.set_scaler(filepath)

    def set_medium_scaler(self, filepath=""):
        self._medium_tp_model.set_scaler(filepath)
    
    def set_high_scaler(self, filepath=""):
        self._high_tp_model.set_scaler(filepath)

    def set_classifier_scaler(self, filepath=""):
        self._classifier.set_scaler(filepath)

    def set_test(self, x, y):
        self._test_x = x
        self._test_y = y

    def write_datasets_to_csv(self, predicted, input=None, true=None, filename=None):
        if not filename:
            filename = self._model_name
        model_output_path = config["global"]["MODEL_OUTPUT_PATH"]
        if model_output_path[-1] not in ["\\", "/"]:
            model_output_path += "/"
        df = pd.DataFrame()
        if not input:
            x = self._test_x.squeeze()
        else:
            x = input.squeeze()
        if not true:
            true = pd.Series(self._test_y.squeeze().tolist())
        else:
            true = pd.Series(true.squeeze().tolist())
        predicted = pd.Series(predicted.squeeze().tolist())
        x_list = x.tolist()
        input_values = x[:,-1]
        df["input_sequence"] = pd.Series(x_list)
        df["input_value"] = input_values
        df["true"] = true
        df["predicted"] = predicted
        df.to_csv(model_output_path+"{}.csv".format(filename), encoding="utf-8", index=False)
