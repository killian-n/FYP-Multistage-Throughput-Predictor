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


class SingleSelectionMultistagePredictor(tf.keras.Model):
    def __init__(self, raw_data, **kwargs):
        super().__init__(**kwargs)
        self._raw_data = raw_data
        self._train_labels = None
        self._test_labels = None
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
        self._preprocessor = DataPreProcessor(self._raw_data, include_features=include_features, predict=predict,
         use_predict=use_predict, manual_mode=manual_mode, scaler=scaler, scaler_file_name=scaler_file_name)

    def build_and_train(self, epochs=10, batch_size=100, validation_split=0.2):
        self._label_predictor = LabelPredictor(model_name="SSMSP_label_predictor")
        self._label_predictor.pre_process(preprocessor=self._preprocessor)
        self._label_predictor.build_model()
        self._label_predictor.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._low_tp_model = SimpleLSTM(model_name="SSMSP_low")
        self._low_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_low_train_sequences(), test=self._preprocessor.get_low_test_sequences())
        self._low_tp_model.build_model()
        self._low_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._medium_tp_model = SimpleLSTM(model_name="SSMSP_medium")
        self._medium_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_medium_train_sequences(), test=self._preprocessor.get_medium_test_sequences())
        self._medium_tp_model.build_model()
        self._medium_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

        self._high_tp_model = SimpleLSTM(model_name="SSMSP_high")
        self._high_tp_model.pre_process(preprocessed=True, train=self._preprocessor.get_high_train_sequences(), test=self._preprocessor.get_high_test_sequences())
        self._high_tp_model.build_model()
        self._high_tp_model.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def load_models(self):
        self._label_predictor = tf.keras.models.load_model("src/saved.objects/SSMSP_label_predictor.hdf5")
        x = [[-0.5643563,-0.5267904,-0.62764394,-0.6770542,-0.43689725,-0.5831835 ,-0.52086824,-0.5683339 ,-0.638516 ,-0.5831835 ]]
        x = np.array(x)
        y = self._label_predictor.predict(x)
        print(y)
        # self._low_tp_model = tf.keras.models.load_model("src/saved.objects/SSMSP_low.hdf5")
        # self._medium_tp_model = tf.keras.models.load_model("src/saved.objects/SSMSP_medium.hdf5")
        # self._high_tp_model = tf.keras.models.load_model("src/saved.objects/SSMSP_high.hdf5")

    def test(self):
        self._label_predictor.test()
        self._low_tp_model.test()
        self._medium_tp_model.test()
        self._high_tp_model.test()
        print("we got this far")
        pass

    def predict(self, x_squences, batch_size=10):
        pass

    def get_preformance_metrics(self):
        pass


if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = SingleSelectionMultistagePredictor(raw_data)
    # example.pre_process()
    # example.build_and_train()
    # example.test()
    example.load_models()