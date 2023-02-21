import configparser
import sys
import pandas as pd
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from models.single_selection_multistage_predictor import SingleSelectionMultistagePredictor


# Change name to multistage all

class MultiSelectionMultistagePredictor(SingleSelectionMultistagePredictor):
    def __init__(self, raw_data=pd.DataFrame(), preprocessor=None, model_name="MSMSP", loss="categorical_crossentropy"):
        super().__init__(raw_data, preprocessor=preprocessor)
        self._model_name = model_name
        self._loss = loss

    def __call__(self, x_sequences):
        label = self._label_predictor(x_sequences).numpy()

        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        low_result = self._low_tp_model(x_sequences).numpy()*low_labels
        medium_result = self._medium_tp_model(x_sequences).numpy()*med_labels
        high_result = self._high_tp_model(x_sequences).numpy()*high_labels
        result = low_result + medium_result + high_result
        return result
    
if __name__ == "__main__":
    raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
    example = MultiSelectionMultistagePredictor(raw_data)
    example.pre_process()
    example.build_and_train()
    example.test()
    b = example.get_performance_metrics()
    print(b)
