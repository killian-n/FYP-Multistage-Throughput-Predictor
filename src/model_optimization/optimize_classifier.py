import configparser
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
from keras_tuner import RandomSearch
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor

raw_data = pd.read_csv(config["global"]["PROJECT_PATH"]+"Datasets/Raw/all_4G_data.csv", encoding="utf-8")
preprocessor = DataPreProcessor(raw_data, scaler_file_name="tuning_scaler.sav", include_features=["NRxRSRQ", "RSRQ","RSRP" ,"SNR", "CQI", "RSSI", "NRxRSRP"])

train_x, train_y = preprocessor.get_label_predictor_train()
test_x, test_y = preprocessor.get_label_predictor_test()

# Define the model-building function
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hp.Int('classifier_input', min_value=32, max_value=128, step=32), 
                          return_sequences=True, input_shape=(train_x.shape[1:])))
    model.add(tf.keras.layers.Dropout(hp.Float('input_layer_dropout', min_value=0.0, max_value=0.5, step=0.1)))
    # Tune the number of hidden LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 2)):
        model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=96, step=32), 
                              return_sequences=True))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Flatten())
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=32, max_value=96, step=32),
                            activation=hp.Choice('dense_activation_' + str(i), values=['relu', 'sigmoid'])))
        # Tune the inclusion of dropout layers
        model.add(tf.keras.layers.Dropout(hp.Float('dense_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
                    
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model




# Define the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=7,
    executions_per_trial=2,
    directory='model_tuning',
    project_name='classifier_2_tuning'
)

# Start the tuning process
tuner.search(train_x, train_y, epochs=50, validation_data=(test_x, test_y))

# Print the best model summary
best_model = tuner.get_best_models(num_models=1)[0]
best_model.build(input_shape=(train_x.shape[1:]))
best_model.summary()
