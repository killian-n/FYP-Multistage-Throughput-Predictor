import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from keras.utils.layer_utils import count_params
import csv
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.multistage_regression_model import MultiStageLSTM
from training_models.classifier import ThroughputClassifier
from data_transformation.preprocessor import DataPreProcessor


# Change name to multistage_one

class SingleSelectionMultistagePredictor:
    def __init__(self, raw_data=pd.DataFrame(), preprocessor=None, model_name="SSMSP", loss="sparse_categorical_crossentropy"):
        self._preprocessor = preprocessor
        self._loss = loss
        self._model_name = model_name
        self._raw_data = raw_data
        self._test_x = None
        self._test_y = None
        self._label_predictor = None
        self._low_tp_model = None
        self._medium_tp_model = None
        self._high_tp_model = None
        self._input_shape = ()
        self._output_shape = ()
        self._is_scaled = None
        if self._preprocessor:
            self._is_scaled = self._preprocessor.is_scaled()
        self._results = []

    def predict(self, x_sequences, no_of_batches=1000):
        if no_of_batches > x_sequences.shape[0]:
            no_of_batches = 1
        predictions = np.zeros((x_sequences.shape[0], self._output_shape[1]))
        index = 0
        for arr in np.array_split(x_sequences, no_of_batches):
            result = self.__call__(arr)
            predictions[index:result.shape[0]+index, :] = result
            index += result.shape[0]
        return predictions
    
    def __call__(self, x_sequences):
        label = self._label_predictor(x_sequences).numpy()
        
        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        x_sequences = self.__inverse_scale(x_sequences)

        low_result = self._low_tp_model(x_sequences)*low_labels
        medium_result = self._medium_tp_model(x_sequences)*med_labels
        high_result = self._high_tp_model(x_sequences)*high_labels
        result = low_result + medium_result + high_result
        return result

    def predict(self, x_sequences, no_of_batches=1000):
        """ """
        if no_of_batches > x_sequences.shape[0]:
            no_of_batches = 1
        predictions = np.zeros((x_sequences.shape[0], self._output_shape[1]))
        index = 0
        for arr in np.array_split(x_sequences, no_of_batches):
            result = self.__call__(arr)
            predictions[index:result.shape[0]+index, :] = result
            index += result.shape[0]
        return predictions

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler_file_name="SSMSP_scaler.sav"):
        if self._loss == "sparse_categorical_crossentropy":
            sparse=True
        else:
            sparse=False
        if not self._preprocessor:
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
                use_predict=use_predict, manual_mode=manual_mode, scaler_file_name=scaler_file_name)
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()
        self._test_y = self.__inverse_scale(self._test_y, is_x=False)

    def build_and_train(self, epochs=70, batch_size=100, validation_split=0.2):

        if self._loss == "sparse_categorical_crossentropy":
            sparse=True
        else:
            sparse=False
        self._label_predictor = ThroughputClassifier(model_name=self._model_name+"_classifier", sparse=sparse, preprocessor=self._preprocessor)
        self._label_predictor.pre_process()
        self._label_predictor.build_model(loss=self._loss)
        self._label_predictor.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        
        x_train, y_train = self._preprocessor.get_low_train_sequences()
        x_train = self.__inverse_scale(x_train)
        print("Sample of x_train",x_train[0])
        y_train = self.__inverse_scale(y_train, is_x=False)
        print("Sample of y_train",y_train[0])
        x_test, y_test = self._preprocessor.get_low_test_sequences()
        x_test = self.__inverse_scale(x_test)
        y_test = self.__inverse_scale(y_test, is_x=False)
        
        self._low_tp_model = MultiStageLSTM(model_name="{}_low".format(self._model_name), preprocessor=self._preprocessor)
        self._low_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._low_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._low_tp_model.build_model()
        self._low_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        x_train, y_train = self._preprocessor.get_medium_train_sequences()
        x_train = self.__inverse_scale(x_train)
        y_train = self.__inverse_scale(y_train, is_x=False)
        x_test, y_test = self._preprocessor.get_medium_test_sequences()
        x_test = self.__inverse_scale(x_test)
        y_test = self.__inverse_scale(y_test, is_x=False)

        self._medium_tp_model = MultiStageLSTM(model_name="{}_medium".format(self._model_name), preprocessor=self._preprocessor)
        self._medium_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._medium_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._medium_tp_model.build_model()
        self._medium_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        x_train, y_train = self._preprocessor.get_high_train_sequences()
        x_train = self.__inverse_scale(x_train)
        y_train = self.__inverse_scale(y_train, is_x=False)
        x_test, y_test = self._preprocessor.get_high_test_sequences()
        x_test = self.__inverse_scale(x_test)
        y_test = self.__inverse_scale(y_test, is_x=False)

        self._high_tp_model = MultiStageLSTM(model_name="{}_high".format(self._model_name), preprocessor=self._preprocessor)
        self._high_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._high_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._high_tp_model.build_model()
        self._high_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._input_shape = self._label_predictor.get_input_shape()
        self._output_shape = self._high_tp_model.get_output_shape()
        
    def load_models(self):
        self._label_predictor = ThroughputClassifier(model_name=self._model_name+"_label_predictor")
        label_predictor = tf.keras.models.load_model("src/saved.objects/{}_label_predictor.hdf5".format(self._model_name))
        self._label_predictor.set_model(label_predictor)

        self._low_tp_model = MultiStageLSTM(model_name="{}_low".format(self._model_name))
        low_tp_model = tf.keras.models.load_model("src/saved.objects/{}_low.hdf5".format(self._model_name))
        self._low_tp_model.set_model(low_tp_model)

        self._medium_tp_model = MultiStageLSTM(model_name="{}_medium".format(self._model_name))
        medium_tp_model = tf.keras.models.load_model("src/saved.objects/{}_medium.hdf5".format(self._model_name))
        self._medium_tp_model.set_model(medium_tp_model)

        self._high_tp_model = MultiStageLSTM(model_name="{}_high".format(self._model_name))
        high_tp_model = tf.keras.models.load_model("src/saved.objects/{}_high.hdf5".format(self._model_name))
        self._high_tp_model.set_model(high_tp_model)

        self._input_shape = self._label_predictor.get_input_shape()
        self._output_shape = self._high_tp_model.get_output_shape()

    def get_mse(self, true, predicted):
        mse = np.power(np.squeeze(true)-predicted, 2).mean()
        return mse

    def get_average_bias(self, true, predicted):
        average_bias = np.mean(np.squeeze(true)-predicted)
        return average_bias

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
        train_time = 0
        trainable_params = 0
        non_trainable_params = 0
        model_size = 0

        # Individual Tests
        self._label_predictor.test()
        self._low_tp_model.test()
        self._medium_tp_model.test()
        self._high_tp_model.test()
        
        train_time += self._label_predictor.get_performance_metrics()[3]
        model_size += self._label_predictor.get_model_size()
        trainable_params += count_params(self._label_predictor.get_model().trainable_weights)
        non_trainable_params += count_params(self._label_predictor.get_model().non_trainable_weights)

        train_time += self._low_tp_model.get_performance_metrics()[3]
        model_size += self._low_tp_model.get_model_size()
        trainable_params += count_params(self._low_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._low_tp_model.get_model().non_trainable_weights)
        
        train_time += self._medium_tp_model.get_performance_metrics()[3]
        model_size += self._medium_tp_model.get_model_size()
        trainable_params += count_params(self._medium_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._medium_tp_model.get_model().non_trainable_weights)

        train_time += self._high_tp_model.get_performance_metrics()[3]
        model_size += self._high_tp_model.get_model_size()
        trainable_params += count_params(self._high_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._high_tp_model.get_model().non_trainable_weights)

        predict_start = time()
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        
        mse = self.get_mse(self._test_y, predicted_y)
        mae = self.get_mae(self._test_y, predicted_y)
        average_bias = self.get_average_bias(self._test_y, predicted_y)
        mape = self.get_mape(self._test_y, predicted_y)
        self._results = [self._model_name, trainable_params, non_trainable_params, train_time, time_to_predict, mse, mae, average_bias, mape, model_size]
        self.write_to_csv()
        self.save_output(self._test_x, self._model_name+"_test_x")
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")

    def get_performance_metrics(self):
        return self._results
    
    def __inverse_scale(self, input_array, is_x=True):
        input_shape = input_array.shape
        if is_x:
             input_array = self._preprocessor.get_scaler().inverse_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), self._preprocessor.get_scaler().n_features_in_))
            transform[:,0] = input_array
            transform = self._preprocessor.get_scaler().inverse_transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

    def write_to_csv(self):
        csv_file = config["metrics"]["RESULTS_PATH"]
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._results)

    def save_output(self,output,filename="DEFAULT_NAME_OUTPUTS"):
        filename = "Datasets/Model_Outputs/"+filename
        np.save(filename, output)

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    preprocessor = DataPreProcessor(raw_data, scaler_file_name="debugging_scaler.sav", include_features=["NRxRSRQ", "RSRQ", "SNR", "CQI", "RSSI"])
    example = SingleSelectionMultistagePredictor(preprocessor=preprocessor, model_name="multi_1")
    example.pre_process()
    example.build_and_train()
    example.test()
    # b = example.get_performance_metrics()
    # print(b)
    # transform = np.zeros((1, 7))
    # print(transform)
    # epsilon = 50
    # transform[0,0] = epsilon
    # print(transform)


    # a = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # b = [[4,1], [1,5]]
    # a = np.array(a)

    # # a[0:2, :] = np.array([[1,1,1,1], [1,1,1,1]])
    # a = [[[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [2, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0]]]
    # a = np.array(a)
    # # print(a)
    # b = a.reshape((5,4,2))
    # print(a.T.flatten())

    # print(b[:,:,0])
    # for arr in b:
    #     print(arr)
    #     print("====")
    # print(b)
    # print(b.shape)

    # def test_fuc(a):
    #     print(a)
    #     print(a.shape)
    #     print("======")
    #     b = a
    #     return b

    # f = np.vectorize(test_fuc, signature="(n,m)->(n,m)")
    # k = f(b)
    # print(type(k))
    # np.apply_along_axis(test_fuc, 1, a)
    # b = np.array(b)
    # c = np.mean((a-b)**2)
    # d = np.power(a-b, 2).mean()
    # print(a.shape)
    # mse = tf.keras.losses.mean_squared_error(a, b)
    # print("HERE\n", np.mean(mse.numpy()), "\n=========")
    # print("HERE\n", c, "\n=========")
    # print("HERE\n", d, "\n=========")

    # example.load_models()
    # a = np.array([[.9, .9, .76, .8, 1.2, 1, .95, .76, .77, .81], [-1.2, -1.2, -1.26, -1.5, -1.1, -1.3, -1.1, -1.4, -1, -.7]])
    # r = example(a)
    # print(r)
    # print(r.shape)
    # print("\n\n===============\n\n")
    # k = example.predict(a)
    # print(k)
    # print(k.shape)
    # a = np.random.rand(201, 10)
    # print("\n\n===============\n\n")
    # k = example.predict(a)
    # print(k)
    # print(k.shape)
    

