import configparser
import sys
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.layer_utils import count_params
from time import time
import pickle
import numpy as np
import csv
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.model_framework import ModelFramework
from data_transformation.preprocessor import DataPreProcessor
from helper_functions.timer import TimingCallback

class ThroughputClassifier(ModelFramework):
    def __init__(self, raw_data=pd.DataFrame(), preprocessor=None, model_name="univariate_classifier", sparse=False, univariate=True):
        super().__init__()
        self._raw_data = raw_data
        self._model_name = model_name
        self._sparse = sparse
        self._preprocessor = preprocessor
        self._univariate = univariate

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, history=10, horizon=5, scale_data=True):
        if not self._preprocessor:
            scaler_file_name = self._model_name + ".sav"
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
             use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name, history=history, horizon=horizon, scale_data=scale_data)

        # Basic formatting
        self._train_x, self._train_y = self._preprocessor.get_label_predictor_train(sparse=self._sparse)
        self._test_x, self._test_y = self._preprocessor.get_label_predictor_test(sparse=self._sparse)
        if self._sparse:
            self._train_y = self._train_y.T
            self._test_y = self._test_y.T

    def __call__(self, inputs):
        if self._univariate:
            return self._model(inputs[:,:,0].reshape((inputs.shape[0],inputs.shape[1], 1)))
        return self._model(inputs)
    
    def predict(self, inputs):
        if self._univariate:
            return self._model.predict(inputs[:,:,0].reshape((inputs.shape[0],inputs.shape[1], 1)))
        return self._model.predict(inputs)

    def build_model(self, loss="categorical_crossentropy"):
        if self._univariate:
            self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(None, self._train_x.shape[1], 1), return_sequences=True))
            print("in here")
        else:
            self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(self._train_x.shape[1:]), return_sequences=True))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(self._train_x.shape[1:]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64, input_shape=(self._train_x.shape[1:]),return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32, input_shape=(self._train_x.shape[1:]),return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(32))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(3, activation="softmax"))
        self._model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self._model.summary()
        self.set_input_shape()
        self.set_output_shape()

    def train(self, epochs=70, batch_size=32, validation_split=0.2):
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        timer = TimingCallback()
        self._tensorboard = TensorBoard(log_dir="{}src/logs/{}".format(project_path, self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='{}src/saved.objects/{}.hdf5'.format(project_path, self._model_name), verbose = 1, save_best_only=True)
        if self._preprocessor:
            self._class_weights = self._preprocessor.get_class_weights()

        print(self._univariate)
        if self._univariate:
            print("shape", self._train_x[:,:,0].reshape((self._train_x.shape[0],self._train_x.shape[1], 1)).shape)
            self._model.fit(self._train_x[:,:,0].reshape((self._train_x.shape[0],self._train_x.shape[1], 1)), self._train_y, epochs=epochs, batch_size=batch_size,
                validation_split=validation_split, verbose=1,class_weight=self._class_weights, callbacks=[self._checkpointer, self._tensorboard, timer])
        else:
            self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size,
                validation_split=validation_split, verbose=1,class_weight=self._class_weights, callbacks=[self._checkpointer, self._tensorboard, timer])
        self._train_time = sum(timer.logs)

    def test(self):
        trainable_params = count_params(self._model.trainable_weights)
        non_trainable_params = count_params(self._model.non_trainable_weights)
        predict_start = time()
        predicted_y_prob = self.predict(self._test_x)
        predicted_y = np.zeros_like(predicted_y_prob)
        predicted_y[np.arange(len(predicted_y_prob)), predicted_y_prob.argmax(axis=1)] = 1
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        if self._univariate:
            test = self._model.evaluate(self._test_x[:,:,0].reshape((self._test_x.shape[0],self._test_x.shape[1], 1)), self._test_y, batch_size=100)
        else:
            test = self._model.evaluate(self._test_x, self._test_y, batch_size=100)
        accuracy = test[1]
        model_size = self.get_model_size()
        self._results = [self._model_name, trainable_params, non_trainable_params, self._train_time, time_to_predict, "n/a", "n/a", "n/a", "n/a",model_size, accuracy]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")

    def scale(self, input_array):
        input_shape = input_array.shape
        if not self._scaler:
            self._scaler = self._preprocessor.get_scaler()
        input_array = self._scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        return input_array

    def set_train(self, train_x, train_y):
        self._train_x = self.scale(train_x)
        self._train_y = train_y        
        if self._sparse:
            self._train_y = self._train_y.T

    def set_test(self, test_x, test_y):
        self._test_x = self.scale(test_x)
        self._test_y = test_y
        if self._sparse:
            self._test_y = self._test_y.T

    def get_performance_metrics(self):
        return self._results

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)

    def set_class_weights(self, filename="", set_as_none=False):
        if set_as_none:
            self._class_weights=None
        else:
            if not filename:
                filename = "class_weights"
            saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
            if saved_objects_path[-1] not in ["\\", "/"]:
                saved_objects_path += "/"
            filepath = saved_objects_path+filename
            self._class_weights = pickle.load(open(filepath, "rb"))



if __name__ == "__main__":
    # {0: 1.7086360253729758, 1: 1.1781551618814905, 2: 0.6385886840432295}
    raw_data = pd.read_csv("{}Datasets/Raw/all_4G_data.csv", encoding="utf-8".format(config["global"]["PROJECT_PATH"]))
    preprocessor_univariate = DataPreProcessor(raw_data, scaler_file_name="throw_away_univariate.sav")
    example = ThroughputClassifier(model_name="label_preditor_univariate_unweighted")
    example.pre_process(preprocessor=preprocessor_univariate)
    example.build_model()
    example.train()
    example.test()
