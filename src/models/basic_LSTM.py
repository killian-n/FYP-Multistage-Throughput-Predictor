import configparser
import sys
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.preprocessor import DataPreProcessor

# Read in dataset and scaler
raw_data = pd.read_csv("Datasets/Raw/all_4G_data.csv", encoding="utf-8")

pre_processor = DataPreProcessor(dataframe=raw_data)

train_x, train_y = pre_processor.get_train_sequences()
test_x, test_y = pre_processor.get_test_sequences()
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation="relu", input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, activation="relu",return_sequences=False))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(train_y.shape[1]))
model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
model.summary()

checkpointer = ModelCheckpoint(filepath='src/saved.objects/univariate.1.hdf5', verbose = 1, save_best_only=True)
model.fit(train_x, train_y, epochs=10, batch_size=100, validation_split=0.2, verbose=1, callbacks=[checkpointer])



