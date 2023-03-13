import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from keras_tuner import RandomSearch
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor

raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")
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

train_x, train_y = global_preprocessor.get_low_train_sequences()
test_x, test_y = global_preprocessor.get_low_test_sequences()

train_x = inverse_scale(preprocessor=global_preprocessor, input_array=train_x)
train_y = inverse_scale(preprocessor=global_preprocessor, input_array=train_y, is_x=False)
test_x = inverse_scale(preprocessor=global_preprocessor, input_array=test_x)
test_y = inverse_scale(preprocessor=global_preprocessor, input_array=test_y, is_x=False)

train_x, low_scaler = scale(train_x, is_x_train=True)
train_y, low_scaler = scale(train_y, low_scaler, is_x=False)
test_x, low_scaler  = scale(test_x, scaler=low_scaler)
test_y, low_scaler  = scale(test_y, scaler=low_scaler, is_x=False)
 
print(train_x.shape)
print(train_x[0])
print(train_y.shape)
print(train_y[0])

# Define the model-building function
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hp.Int('low_input', min_value=64, max_value=128, step=32), 
                          return_sequences=True, input_shape=(train_x.shape[1:])))
    model.add(tf.keras.layers.Dropout(hp.Float('input_layer_dropout', min_value=0.0, max_value=0.5, step=0.1)))
    # Tune the number of hidden LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=64, step=16), 
                              return_sequences=True))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    # Flatten output to pass into the dense layer
    model.add(tf.keras.layers.Flatten())
    # Dense Layers
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=16, max_value=32, step=8),
                            activation=hp.Choice('dense_activation_' + str(i), values=['relu', 'sigmoid'])))
        # Tune the inclusion of dropout layers
        model.add(tf.keras.layers.Dropout(hp.Float('dense_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    # Output Layer is fixed                    
    model.add(tf.keras.layers.Dense(train_y.shape[1]))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse')
    return model

# Define the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=8,
    executions_per_trial=3,
    directory='model_tuning',
    project_name='low_regresssion_tuning'
)

# Start the tuning process
tuner.search(train_x, train_y, epochs=70, validation_data=(test_x, test_y))

# Print the best model summary
best_model = tuner.get_best_models(num_models=1)[0]
best_model.build(input_shape=(train_x.shape[1:]))
best_model.summary()
