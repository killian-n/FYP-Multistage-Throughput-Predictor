import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras_tuner import RandomSearch
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor

raw_data = pd.read_csv(config["global"]["PROJECT_PATH"]+"Datasets/Raw/all_4G_data.csv", encoding="utf-8")
global_preprocessor = DataPreProcessor(raw_data, scaler_file_name="tuning_scaler.sav", include_features=["NRxRSRQ", "RSRQ","RSRP" ,"SNR", "CQI", "RSSI", "NRxRSRP"])
# global_preprocessor = DataPreProcessor(raw_data, scaler_file_name="tuning_scaler.sav")



def inverse_scale(preprocessor, input_array, is_x=True):
    input_shape = input_array.shape
    if is_x:
            input_array = preprocessor.get_scaler().inverse_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
    else:
        input_array = np.squeeze(input_array).flatten()
        transform = np.zeros((len(input_array), preprocessor.get_scaler().n_features_in_))
        transform[:,0] = input_array
        transform = preprocessor.get_scaler().inverse_transform(transform)
        input_array = transform[:,0].reshape(input_shape)
    return input_array

def scale(input_array, scaler=None, is_x=True, is_x_train=False):
    input_shape = input_array.shape
    if is_x:
        if is_x_train:
            scaler = MinMaxScaler((-1, 1))
            input_array = scaler.fit_transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
    else:
        input_array = np.squeeze(input_array).flatten()
        transform = np.zeros((len(input_array), scaler.n_features_in_))
        transform[:,0] = input_array
        transform = scaler.transform(transform)
        input_array = transform[:,0].reshape(input_shape)
    return input_array, scaler

train_x, train_y = global_preprocessor.get_train_sequences()
test_x, test_y = global_preprocessor.get_test_sequences()


# Define the model-building function
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hp.Int('baseline_input', min_value=128, max_value=256, step=32), 
                          return_sequences=True, input_shape=(train_x.shape[1:])))
    model.add(tf.keras.layers.Dropout(hp.Float('input_layer_dropout', min_value=0.0, max_value=0.5, step=0.1)))
    # Tune the number of hidden LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 2)):
        model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=64, max_value=128, step=16), 
                              return_sequences=True))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Flatten())
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=32, max_value=64, step=8),
                            activation=hp.Choice('dense_activation_' + str(i), values=['relu', 'sigmoid'])))
        # Tune the inclusion of dropout layers
        model.add(tf.keras.layers.Dropout(hp.Float('dense_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
                    
    model.add(tf.keras.layers.Dense(train_y.shape[1]))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse')
    
    return model

# Define the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=7,
    executions_per_trial=2,
    directory='model_tuning',
    project_name='baseline_regresssion_tuning'
)

# Start the tuning process
tuner.search(train_x, train_y, epochs=70, validation_data=(test_x, test_y))

# Print the best model summary
best_model = tuner.get_best_models(num_models=1)[0]
best_model.build(input_shape=(train_x.shape[1:]))
best_model.summary()