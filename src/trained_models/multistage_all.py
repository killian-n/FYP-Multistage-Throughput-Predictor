import configparser
import sys
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.multistage_one import MultistageOne

class MultistageAll(MultistageOne):
    def __init__(self, model_name="multiAll"):
        super().__init__(model_name=model_name)

    def __call__(self, x_sequences):
        label = self._classifier(x_sequences).numpy()

        low_labels = label[:,0].reshape((label.shape[0],1))
        med_labels = label[:,1].reshape((label.shape[0],1))
        high_labels = label[:,2].reshape((label.shape[0],1))

        low_result = self._low_tp_model(x_sequences)*low_labels
        medium_result = self._medium_tp_model(x_sequences)*med_labels
        high_result = self._high_tp_model(x_sequences)*high_labels
        result = low_result + medium_result + high_result
        return result