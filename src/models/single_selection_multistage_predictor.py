import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from keras.utils.layer_utils import count_params
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.simple_LSTM import SimpleLSTM
from models.label_predictor import LabelPredictor
from data_transformation.preprocessor import DataPreProcessor


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

        self._results = []

    def __model_selector(self, x_sequence):
        try:
            x_sequence = np.reshape(x_sequence, (1,self._input_shape[1],self._input_shape[2]))
            label = self._label_predictor(x_sequence)
            if np.argmax(label) == 0:
                result = self._low_tp_model(x_sequence)
            elif np.argmax(label) == 1:
                result = self._medium_tp_model(x_sequence)
            else:
                result = self._high_tp_model(x_sequence)
            return result.numpy()
        except:
            print("X:", x_sequence)
            print("X shape:", x_sequence.shape)
            print("Desired shape: (",1, self._input_shape[1], self._input_shape[2], ")")
            sys.exit()

    def quick_predict(self, x_sequences, no_of_batches=1000):
        if no_of_batches > x_sequences.shape[0]:
            no_of_batches = 1
        print("OUTPUT SHAPE", self._output_shape)
        predictions = np.zeros((x_sequences.shape[0], self._output_shape[1]))
        index = 0
        for arr in np.array_split(x_sequences, no_of_batches):
            result = self.__call__(arr)
            predictions[index:result.shape[0]+index, :] = result
            index += result.shape[0]
        return predictions
    
    def quick_call(self, x_sequences):
        """Running the sequences through all 3 models is quicker than using
        __model_selector due to parallelisation."""
        label = self._label_predictor(x_sequences).numpy()
        
        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        low_result = self._low_tp_model(x_sequences).numpy()*low_labels
        medium_result = self._medium_tp_model(x_sequences).numpy()*med_labels
        high_result = self._high_tp_model(x_sequences).numpy()*high_labels
        result = low_result + medium_result + high_result
        return result


    def __call__(self, x_sequences):
        try:
            get_prediction = np.vectorize(self.__model_selector, signature="(n,m)->(j, k)")
            results = get_prediction(x_sequences)
            if len(self._output_shape) < 3:
                results = results.reshape((results.shape[0], self._output_shape[1]))
            else:
                results = results.reshape((results.shape[0], self._output_shape[1], self._output_shape[2]))
            return results
        except Exception as e:
            print(x_sequences[0:10])
            print(x_sequences.shape)
            print("\n\n=========\n", e)
            with open("Datasets/debug.txt", "w") as f:
                f.write(np.array2string(x_sequences, precision=2, separator=","))
            sys.exit()

    def predict(self, x_sequences, no_of_batches=1000):
        if no_of_batches > x_sequences.shape[0]:
            no_of_batches = 1
        print("OUTPUT SHAPE", self._output_shape)
        predictions = np.zeros((x_sequences.shape[0], self._output_shape[1]))
        index = 0
        for arr in np.array_split(x_sequences, no_of_batches):
            result = self.__call__(arr)
            predictions[index:result.shape[0]+index, :] = result
            index += result.shape[0]
        return predictions

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="SSMSP_scaler.sav"):
        if not self._preprocessor:
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
                use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()

    def build_and_train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._label_predictor = LabelPredictor(model_name=self._model_name+"_label_predictor")
        self._label_predictor.pre_process(preprocessor=self._preprocessor)
        self._label_predictor.build_model(loss=self._loss)
        self._label_predictor.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._low_tp_model = SimpleLSTM(model_name="{}_low".format(self._model_name))
        self._low_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_low_train_sequences(), test=self._preprocessor.get_low_test_sequences())
        self._low_tp_model.build_model()
        self._low_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._medium_tp_model = SimpleLSTM(model_name="{}_medium".format(self._model_name))
        self._medium_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_medium_train_sequences(), test=self._preprocessor.get_medium_test_sequences())
        self._medium_tp_model.build_model()
        self._medium_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._high_tp_model = SimpleLSTM(model_name="{}_high".format(self._model_name))
        self._high_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_high_train_sequences(), test=self._preprocessor.get_high_test_sequences())
        self._high_tp_model.build_model()
        self._high_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._input_shape = self._label_predictor.get_input_shape()
        self._output_shape = self._high_tp_model.get_output_shape()
        
    def load_models(self):
        self._label_predictor = LabelPredictor(model_name=self._model_name+"_label_predictor")
        label_predictor = tf.keras.models.load_model("src/saved.objects/{}_label_predictor.hdf5".format(self._model_name))
        self._label_predictor.set_model(label_predictor)

        self._low_tp_model = SimpleLSTM(model_name="{}_low".format(self._model_name))
        low_tp_model = tf.keras.models.load_model("src/saved.objects/{}_low.hdf5".format(self._model_name))
        self._low_tp_model.set_model(low_tp_model)

        self._medium_tp_model = SimpleLSTM(model_name="{}_medium".format(self._model_name))
        medium_tp_model = tf.keras.models.load_model("src/saved.objects/{}_medium.hdf5".format(self._model_name))
        self._medium_tp_model.set_model(medium_tp_model)

        self._high_tp_model = SimpleLSTM(model_name="{}_high".format(self._model_name))
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

    def test(self):
        train_time = 0
        trainable_params = 0
        non_trainable_params = 0
        
        train_time += self._label_predictor.get_performance_metrics()
        trainable_params += count_params(self._label_predictor.get_model().trainable_weights)
        non_trainable_params += count_params(self._label_predictor.get_model().non_trainable_weights)

        train_time += self._low_tp_model.get_performance_metrics()
        trainable_params += count_params(self._low_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._low_tp_model.get_model().non_trainable_weights)
        
        train_time += self._medium_tp_model.get_performance_metrics()
        trainable_params += count_params(self._medium_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._medium_tp_model.get_model().non_trainable_weights)

        train_time += self._high_tp_model.get_performance_metrics()
        trainable_params += count_params(self._high_tp_model.get_model().trainable_weights)
        non_trainable_params += count_params(self._high_tp_model.get_model().non_trainable_weights)

        # Individual Tests
        # self._label_predictor.test()
        # self._low_tp_model.test()
        # self._medium_tp_model.test()
        # self._high_tp_model.test()
        predict_start = time()
        predicted_y = self.quick_predict(self._test_x)
        time_to_predict = time()-predict_start
        ## TOO BIG TO FIT IN MEMORY 
        ## WRITE FUNCTION TO COMPUTE PIECE WISE
        mse = self.get_mse(self._test_y, predicted_y)
        mae = self.get_mae(self._test_y, predicted_y)
        average_bias = self.get_average_bias(self._test_y, predicted_y)
        self._results = [self._model_name, trainable_params, non_trainable_params, train_time, time_to_predict, mse, mae, average_bias, self._test_y.shape[0]]


    def get_performance_metrics(self):
        return self._results

if __name__ == "__main__":
    # raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    # example = SingleSelectionMultistagePredictor(raw_data)
    # example.pre_process()
    # example.build_and_train()
    # example.test()

    # a = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # b = [[4,1], [1,5]]
    # a = np.array(a)

    # # a[0:2, :] = np.array([[1,1,1,1], [1,1,1,1]])
    a = [[[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], [[1, 0, 0, 0], [2, 0, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 0]]]
    a = np.array(a)
    # print(a)
    b = a.reshape((5,4,2))
    print(a.T.flatten())

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
    

