import configparser
import sys
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from data_transformation.data_preprocessor import WirelessDataPreProcessor

# Read in dataset and scaler
raw_data = pd.read_csv("Datasets/Transformed/4G_data_ASN.csv", encoding="utf-8")
scaler_used = pickle.load(open("src/saved.objects/scaler_1.sav", "rb"))

# create Preprocessor in manual mode as imputation,
# normalization, one hot encoding and average feature creation already done

pre_processor = WirelessDataPreProcessor(dataframe=raw_data, manual_mode=True)
pre_processor.create_grouped_sequences(exclude_columns=["Timestamp", "CellID", "DL_bitrate", "UL_bitrate"],
y_columns=["Average_DL_bitrate", "Average_UL_bitrate"])

x = pre_processor.get_x_sequences()
y = pre_processor.get_y_sequences()

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42, test_size=0.2)


print(y.shape)
print(x.shape)
print("========")
print(train_x.shape)
print(train_y.shape)




model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, activation="relu", input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, activation="relu",return_sequences=False))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(train_y.shape[1]))
model.compile(optimizer="adam", loss="mse")
model.summary()


