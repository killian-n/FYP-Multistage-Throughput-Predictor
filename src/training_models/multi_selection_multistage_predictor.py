import configparser
import sys
import pandas as pd
import tensorflow as tf
from time import time
from keras.utils.layer_utils import count_params
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from training_models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor
from training_models.classifier import ThroughputClassifier
from training_models.multistage_regression_model import MultiStageLSTM
from data_transformation.preprocessor import DataPreProcessor
from training_models.optimized_models import optimizedClassifierModel, optimizedLowRegressionModel, optimizedMediumRegressionModel, optimizedHighRegressionModel


# Change name to multistage all

class MultiSelectionMultistagePredictor(SingleSelectionMultistagePredictor):
    def __init__(self, raw_data=pd.DataFrame(), preprocessor=None, model_name="MSMSP", loss="categorical_crossentropy", pretrained_sequence_models=""):
        super().__init__(raw_data, preprocessor=preprocessor)
        self._model_name = model_name
        self._loss = loss
        self._train_time = 0
        self._test_time = 0
        self._trainable_params = 0
        self._non_trainable_params = 0
        self._model_size = 0
        self._pretrained_name = ""

        if pretrained_sequence_models:
            self._pretrained_name = pretrained_sequence_models
            self.__load_seq_models()
            results_path = config["metrics"]["RESULTS_PATH"]
            pretrained_results = pd.read_csv(results_path, index_col=None)
            self.__load_pretrained_metrics(pretrained_results, "low")
            self.__load_pretrained_metrics(pretrained_results, "medium")
            self.__load_pretrained_metrics(pretrained_results, "high")
            self.__load_pretrained_metrics(pretrained_results, "classifier")

    def __call__(self, x_sequences):
        label = self._label_predictor(x_sequences).numpy()

        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        x_sequences = self.inverse_scale(x_sequences)

        low_result = self._low_tp_model(x_sequences)*low_labels
        medium_result = self._medium_tp_model(x_sequences)*med_labels
        high_result = self._high_tp_model(x_sequences)*high_labels
        result = low_result + medium_result + high_result
        return result

    def build_and_train(self, epochs=100, batch_size=100, validation_split=0.2):
        sparse=False
        if not self._pretrained_name:
            super().build_and_train()
        else:
            self._label_predictor = optimizedClassifierModel(preprocessor=self._preprocessor, model_name=self._model_name+"_classifier", sparse=sparse)
            self._label_predictor.pre_process()
            self._label_predictor.build_model(loss=self._loss)
            self._label_predictor.train(epochs=epochs, batch_size=batch_size, validation_split=validation_split)
            self._input_shape = self._label_predictor.get_input_shape()

    def test(self):
        if not self._pretrained_name:
            self._low_tp_model.test()
            self._train_time += self._low_tp_model.get_performance_metrics()[3]
            self._model_size += self._low_tp_model.get_model_size()
            self._trainable_params += count_params(self._low_tp_model.get_model().trainable_weights)
            self._non_trainable_params += count_params(self._low_tp_model.get_model().non_trainable_weights)

            self._medium_tp_model.test()
            self._train_time += self._medium_tp_model.get_performance_metrics()[3]
            self._model_size += self._medium_tp_model.get_model_size()
            self._trainable_params += count_params(self._medium_tp_model.get_model().trainable_weights)
            self._non_trainable_params += count_params(self._medium_tp_model.get_model().non_trainable_weights)

            self._high_tp_model.test()
            self._train_time += self._high_tp_model.get_performance_metrics()[3]
            self._model_size += self._high_tp_model.get_model_size()
            self._trainable_params += count_params(self._high_tp_model.get_model().trainable_weights)
            self._non_trainable_params += count_params(self._high_tp_model.get_model().non_trainable_weights)

        self._label_predictor.test()
        self._train_time += self._label_predictor.get_performance_metrics()[3]
        self._model_size += self._label_predictor.get_model_size()
        self._trainable_params += count_params(self._label_predictor.get_model().trainable_weights)
        self._non_trainable_params += count_params(self._label_predictor.get_model().non_trainable_weights)
        predict_start = time()
        # Returns values in true scale
        predicted_y = self.predict(self._test_x)
        time_to_predict = time()-predict_start
        time_to_predict = time_to_predict/self._test_y.shape[0]
        mse = self.get_mse(self._test_y, predicted_y)
        mae = self.get_mae(self._test_y, predicted_y)
        average_bias = self.get_average_bias(self._test_y, predicted_y)
        mape = self.get_mape(self._test_y, predicted_y)
        self._results = [self._model_name, self._trainable_params, self._non_trainable_params, self._train_time, time_to_predict, mse, mae, average_bias, mape, self._model_size]
        self.write_to_csv()
        self.save_output(predicted_y, self._model_name+"_predicted_y")
        self.save_output(self._test_y, self._model_name+"_true_y")


    def __load_pretrained_metrics(self, metrics_df, model_suffix="low"):
            model = metrics_df[metrics_df["model_name"]=="{}_{}".format(self._pretrained_name, model_suffix)]
            model["model_size"] = model["model_size"].astype("float64")
            model["train_time"] = model["train_time"].astype("float64")
            model["trainable_params"] = model["trainable_params"].astype("float64")
            model["non_trainable_params"] = model["non_trainable_params"].astype("float64")
            self._train_time += model["train_time"].iloc[0]
            self._trainable_params += model["trainable_params"].iloc[0]
            self._non_trainable_params += model["non_trainable_params"].iloc[0]
            self._model_size += model["model_size"].iloc[0]


    def __load_seq_models(self):
        project_path = config["global"]["PROJECT_PATH"]
        if project_path[-1] not in ["\\", "/"]:
            project_path += "/"
        x_train, y_train = self._preprocessor.get_low_train_sequences()
        x_train = self.inverse_scale(x_train)
        y_train = self.inverse_scale(y_train, is_x=False)
        x_test, y_test = self._preprocessor.get_low_test_sequences()
        x_test = self.inverse_scale(x_test)
        y_test = self.inverse_scale(y_test, is_x=False)

        self._low_tp_model = MultiStageLSTM(model_name="{}_low".format(self._model_name))
        low_tp_model = tf.keras.models.load_model("{}src/saved.objects/{}_low.hdf5".format(project_path, self._pretrained_name))
        self._low_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._low_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._low_tp_model.set_model(low_tp_model)

        x_train, y_train = self._preprocessor.get_medium_train_sequences()
        x_train = self.inverse_scale(x_train)
        y_train = self.inverse_scale(y_train, is_x=False)
        x_test, y_test = self._preprocessor.get_medium_test_sequences()
        x_test = self.inverse_scale(x_test)
        y_test = self.inverse_scale(y_test, is_x=False)

        self._medium_tp_model = MultiStageLSTM(model_name="{}_medium".format(self._model_name))
        medium_tp_model = tf.keras.models.load_model("{}src/saved.objects/{}_medium.hdf5".format(project_path, self._pretrained_name))
        self._medium_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._medium_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._medium_tp_model.set_model(medium_tp_model)

        x_train, y_train = self._preprocessor.get_high_train_sequences()
        x_train = self.inverse_scale(x_train)
        y_train = self.inverse_scale(y_train, is_x=False)
        x_test, y_test = self._preprocessor.get_high_test_sequences()
        x_test = self.inverse_scale(x_test)
        y_test = self.inverse_scale(y_test, is_x=False)

        self._high_tp_model = MultiStageLSTM(model_name="{}_high".format(self._model_name))
        high_tp_model = tf.keras.models.load_model("{}src/saved.objects/{}_high.hdf5".format(project_path, self._pretrained_name))
        self._high_tp_model.set_train(train_x=x_train, train_y=y_train)
        self._high_tp_model.set_test(test_x=x_test, test_y=y_test)
        self._high_tp_model.set_model(high_tp_model)


        train_x, train_y = self._preprocessor.get_label_predictor_train()
        test_x, test_y = self._preprocessor.get_label_predictor_test()
        self._label_predictor = optimizedClassifierModel(model_name="{}_classifier".format(self._model_name))
        classifier = tf.keras.models.load_model("{}src/saved.objects/{}_high.hdf5".format(project_path,self._pretrained_name))
        self._label_predictor.set_train(train_x, train_y)
        self._label_predictor.set_test(test_x, test_y)
        self._label_predictor.set_model(classifier)

        self._output_shape = self._high_tp_model.get_output_shape()
    
if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    preprocessor_univariate = DataPreProcessor(raw_data, scaler_file_name="univariate.sav")
    multi_all_univariate = MultiSelectionMultistagePredictor(preprocessor=preprocessor_univariate,model_name="unop_univariate_multiAll", pretrained_sequence_models="unop_univariate_multiOne")
    multi_all_univariate.build_and_train()
    multi_all_univariate.test()