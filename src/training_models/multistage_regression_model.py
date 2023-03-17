import configparser
import sys
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from keras.utils.layer_utils import count_params
import numpy as np
from time import time
import pickle

config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.model_framework import ModelFramework
from data_transformation.preprocessor import DataPreProcessor
from helper_functions.timer import TimingCallback

class MultiStageLSTM(ModelFramework):
    def __init__(self, raw_data=pd.DataFrame(),preprocessor=None, model_name="lstm_univariate"):
        super().__init__()
        self._raw_data = raw_data
        self._model_name = model_name
        self._preprocessor = preprocessor
        self._scaler = None
        if self._preprocessor:
            self._is_scaled = self._preprocessor.is_scaled()

    def __call__(self, inputs):
        inputs = self.scale(inputs)
        predictions = self._model(inputs)
        predictions = self.inverse_scale_predictions(predictions)
        return predictions

    def predict(self, inputs, test=False):
        if not test:
            inputs = self.scale(inputs)
        predictions = self._model.predict(inputs)
        print("In {}'s predict, before scaling the prediction is", predictions[0])
        predictions = self.inverse_scale_predictions(predictions)
        return predictions

    def inverse_scale_predictions(self, results):
        results_shape = results.shape
        results = np.squeeze(results).flatten()
        transform = np.zeros((len(results), self._scaler.n_features_in_))
        transform[:,0] = results
        results = self._scaler.inverse_transform(transform)[:,0]
        return results.reshape(results_shape)
    
    def scale(self, input_array, is_x=True, is_x_train=False):
        input_shape = input_array.shape
        if is_x:
            if is_x_train:
                self._scaler = MinMaxScaler((-1, 1))
                input_array = self._scaler.fit_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
                self.save_scaler()
            else:
                input_array = self._scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self._scaler.n_features_in_))
            transform[:,0] = input_array
            transform = self._scaler.transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

    def set_train(self, train_x, train_y):
        self._train_x = train_x
        self._train_y = train_y
        self._train_x = self.scale(self._train_x, is_x_train=True)
        self._train_y = self.scale(self._train_y, is_x=False)

    def set_test(self, test_x, test_y):
        self._test_x = test_x
        self._test_y = test_y
        self._test_x = self.scale(self._test_x)
        self._test_y = self.scale(self._test_y, is_x=False)

    def pre_process(self, include_features=[], predict=["DL_bitrate"],
     use_predict=True, manual_mode=False, scaler=None, history=10, horizon=5, scale_data=False):
        if not self._preprocessor:
            scaler_file_name = self._model_name + ".sav"
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
             use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name, history=history, horizon=horizon, scale_data=False)
        self._train_x, self._train_y = self._preprocessor.get_train_sequences()
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()
        self._train_x = self.scale(self._train_x, is_x_train=True)
        self._test_x = self.scale(self._test_x)
        self._train_y = self.scale(self._train_y, is_x=False)

    def build_model(self):
        epsilon = self.compute_epsilon()
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(256, input_shape=(self._train_x.shape[1], self._train_x.shape[2]), return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128,return_sequences=True))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.compat.v1.keras.layers.CuDNNLSTM(64,return_sequences=False))
        self._model.add(tf.keras.layers.Dropout(.3))
        self._model.add(tf.keras.layers.Dense(self._train_y.shape[1], name="OUT_{}".format(self._model_name)))
        # OPTION 1
        # self._model.compile(optimizer="adam", loss=self.custom_loss(epsilon), metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
        # #####

        # # OPTION 2
        self._model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
        # #####

        self._model.summary()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.set_input_shape()
        self.set_output_shape()

    def train(self, epochs=70, batch_size=100, validation_split=0.2):
        timer = TimingCallback()
        self._tensorboard = TensorBoard(log_dir="src/logs/{}".format(self._model_name))
        self._checkpointer = ModelCheckpoint(filepath='src/saved.objects/{}.hdf5'.format(self._model_name), verbose = 1, save_best_only=False)
        self._model.fit(self._train_x, self._train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
         verbose=1, callbacks=[self._checkpointer, self._tensorboard, timer])
        self._train_time = sum(timer.logs)

    def test(self):
        trainable_params = count_params(self._model.trainable_weights)
        non_trainable_params = count_params(self._model.non_trainable_weights)
        predict_start = time()
        predicted_y = self.predict(self._test_x, test=True)
        test_predicted_y = self._model.predict(self._test_x)
        self.save_output(self.inverse_scale_predictions(test_predicted_y), self._model_name+"COMPARISON")
        print("In {}'s test, predicted y sample is ".format(self._model_name), predicted_y[0])
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        test = self._model.evaluate(self._test_x, self._test_y, batch_size=100)

        # OPTION 1
        # mape, mse, mae = test[0], test[1], test[2]
        #####

        # OPTION 2
        mse, mae = test[0], test[1]
        mape = self.get_mape(self.inverse_scale_predictions(self._test_y), predicted_y)
        #####
        average_bias = self.get_average_bias(self.inverse_scale_predictions(self._test_y), predicted_y)
        model_size = self.get_model_size()
        self._results = [self._model_name, trainable_params, non_trainable_params, self._train_time, time_to_predict, mse, mae, average_bias, mape, model_size]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self.inverse_scale_predictions(self._test_y), self._model_name+"_true_y")

    def compute_epsilon(self, epsilon=50):
        scaler = self._scaler
        transform = np.zeros((1, scaler.n_features_in_))
        transform[0,0] = epsilon
        transform = scaler.transform(transform)
        epsilon = transform[0,0]
        return epsilon

    def get_mape(self, true, predicted, epsilon=50):
        epsilon = self.compute_epsilon(epsilon=epsilon)
        denominator = np.squeeze(true) + epsilon
        try:
            mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
        except Exception as e:
            print(e)
            mape = "n/a"
        return mape

    def get_performance_metrics(self):
        return self._results
        
    def save_scaler(self):
        filename = self._model_name+"_scaler.sav"
        filepath = "src/saved.objects/"+filename
        pickle.dump(self._scaler, open(filepath, "wb"))


    def crazy(self):
        print("SCALED")
        print("x train", self._train_x[0])
        print("y train", self._train_y[0])
        print("x test" , self._test_x[0])
        print("y test", self._test_y[0])
        print("\n\n---------\n", "REMOVING SCALING FROM Y")
        y1 = self.inverse_scale_predictions(np.squeeze(self._train_y))
        y2 = self.inverse_scale_predictions(np.squeeze(self._test_y))
        print("train y", y1[0])
        print("test y", y2[0])




if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    pre_processor = DataPreProcessor(raw_data, manual_mode=False, scaler_file_name="delteete.sav", scale_data=False)

    example = MultiStageLSTM(preprocessor=pre_processor, model_name="Delete_this")
    x, y = pre_processor.get_low_train_sequences()
    example.set_train(x, y)
    print("BEFORE SCALING")
    print("x train", x[0])
    print("y train", y[0])
    x, y = pre_processor.get_low_test_sequences()
    print("x test", x[0])
    print("y test", y[0])
    example.set_test(x, y)
    example.crazy()
    # example.pre_process()
    # example.build_model()
    # example.train(epochs=70)
    # example.test()
