import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.simple_LSTM import SimpleLSTM
from models.label_predictor import LabelPredictor
from data_transformation.preprocessor import DataPreProcessor


class SingleSelectionMultistagePredictor:
    def __init__(self, raw_data, preprocessor=None):
        self._preprocessor = preprocessor
        self._model_name = "SSMSP"
        self._raw_data = raw_data
        self._test_x = None
        self._test_y = None
        self._label_predictor = None
        self._low_tp_model = None
        self._medium_tp_model = None
        self._high_tp_model = None

    def __call__(self, x_sequences):
        label = self._label_predictor(x_sequences)
        if np.argmax(label) == 0:
            result = self._low_tp_model(x_sequences)
        elif np.argmax(label) == 1:
            result = self._medium_tp_model(x_sequences)
        else:
            result = self._high_tp_model(x_sequences)
        return result

    def pre_process(self, include_features=[], predict=["DL_bitrate"], use_predict=True, manual_mode=False, scaler=None, scaler_file_name="SSMSP_scaler.sav"):
        if not self._preprocessor:
            self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
                use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)

    def build_and_train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._label_predictor = LabelPredictor(model_name=self._model_name+"_label_predictor")
        self._label_predictor.pre_process(preprocessor=self._preprocessor)
        self._label_predictor.build_model()
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

    def load_models(self):
        self._label_predictor = tf.keras.models.load_model("src/saved.objects/{}_label_predictor.hdf5".format(self._model_name))
        self._low_tp_model = tf.keras.models.load_model("src/saved.objects/{}_low.hdf5".format(self._model_name))
        self._medium_tp_model = tf.keras.models.load_model("src/saved.objects/{}_medium.hdf5".format(self._model_name))
        self._high_tp_model = tf.keras.models.load_model("src/saved.objects/{}_high.hdf5".format(self._model_name))

    def test(self):
        self._test_x, self._test_y = self._preprocessor.get_test_sequences()
        # Individual Tests
        self._label_predictor.test()
        self._low_tp_model.test()
        self._medium_tp_model.test()
        self._high_tp_model.test()
        print("STARTING TESTING")
        predicted_y = self.__call__(np.array(self._test_x))
        mse = tf.keras.losses.mean_squared_error(np.array(self._test_y), predicted_y).numpy()
        mae = tf.keras.losses.mean_squared_error(np.array(self._test_y), predicted_y).numpy()
        print(self._model_name)
        print("MSE:", mse)
        print("MAE:", mae)
        print("ITS THOSE FUNCTIONS FLOPPING")
        print("===============\n")


    def get_preformance_metrics(self):
        pass

if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = SingleSelectionMultistagePredictor(raw_data)
    example.pre_process()
    example.build_and_train()
    example.test()
