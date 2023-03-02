import configparser
import sys
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
config = configparser.ConfigParser()
config.read('.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)


scaler = pickle.load(open("src/saved.objects/THE_SAVIOR.sav", "rb"))
test_x = np.load("Datasets/Model_Outputs/THE_SAVIOUR_test_x.npy")
train_x = np.squeeze(np.load("Datasets/Model_Outputs/THE_SAVIOUR_train_x.npy"))
true = np.squeeze(np.load("Datasets/Model_Outputs/THE_SAVIOUR_true_y.npy"))
predicted = np.load("Datasets/Model_Outputs/THE_SAVIOUR_predicted_y.npy")
model = tf.keras.models.load_model("src/saved.objects/THE_SAVIOUR.hdf5")


def get_mape(true, predicted, epsilon=50):
    transform = np.zeros((1, scaler._))
    transform[0,0] = epsilon
    transform = scaler.transform(transform)
    epsilon = transform[0,0]
    denominator = np.squeeze(true) + epsilon
    try:
        mape = np.mean(np.abs((np.squeeze(true) - predicted)/denominator))*100
    except Exception as e:
        print(e)
        mape = "n/a"
    return mape

n= 100
test_case = np.reshape(test_x[n,:,:], (1,test_x.shape[1], test_x.shape[2]))
new_predict = model.predict(test_case)
print("new predict",new_predict)
print("old predict", predicted[n])
print("actual", true[n])


transform = np.zeros((5, scaler.n_features_in_))
transform[:,0] = new_predict
print("new unscaled", scaler.inverse_transform(transform)[:,0])

transform = np.zeros((5, scaler.n_features_in_))
transform[:,0] = predicted[n]
print("old predict unscaled", scaler.inverse_transform(transform)[:,0])

transform = np.zeros((5, scaler.n_features_in_))
transform[:,0] = true[n]
print("actual unscaled", scaler.inverse_transform(transform)[:,0])

# print(scaler.inverse_transform(predicted_transform)[:,0])