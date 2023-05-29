import argparse
import configparser
import sys
import numpy as np
import tensorflow as tf
import pickle
from keras_tuner import RandomSearch
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Perform HyperParameter Tuning')

    # Add the arguments to the argument parser
    parser.add_argument('--model', type=str, help='Select which model to train. Values can be one of: [baseline, low, medium , high, classifier]')
    parser.add_argument('--data_prefix', type=str,
                         help="Optional argument. The file prefix of the saved train and test datasets found in Datasets directory.")
    parser.add_argument('--max_search', type=int, help="Optional argument. Default of 20. Maximum no of models to consider.")
    parser.add_argument('--rerun_count', type=int, help="Optional argument. Default of 2. No of times each model is trained and tested for averaging.")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    model_to_train = args.model
    data_prefix = args.data_prefix
    max_search = args.max_search
    rerun_count = args.rerun_count

    if not model_to_train:
        print("No model selected. Training baseline model.")
        parser.print_help()
        sys.exit(1)

    if not model_to_train in ["baseline", "low", "medium", "high", "classifier"]:
        print("Selected model not in provided scope.")
        parser.print_help()
        sys.exit(1)

    if not max_search:
        max_search = 20

    if not rerun_count:
        rerun_count = 2
    
    # Getting Datasets
    train_dir_path = config["global"]["TRAINING_DATASETS_PATH"]
    if train_dir_path[-1] not in ["\\", "/"]:
        train_dir_path += "/"

    train_file_path = "{}{}_{}".format(train_dir_path, data_prefix, model_to_train)
    if model_to_train == "baseline":
        train_file_path = "{}{}".format(train_dir_path, data_prefix)
    train_x = np.load("{}_train_x.npy".format(train_file_path))
    train_y = np.load("{}_train_y.npy".format(train_file_path))

    test_dir_path = config["global"]["TESTING_DATASETS_PATH"]
    if test_dir_path[-1] not in ["\\", "/"]:
        test_dir_path += "/"

    test_file_path = "{}{}_{}".format(test_dir_path, data_prefix, model_to_train)
    if model_to_train == "baseline":
        test_file_path = "{}{}".format(test_dir_path, data_prefix)
    test_x = np.load("{}_test_x.npy".format(test_file_path))
    test_y = np.load("{}_test_y.npy".format(test_file_path))


    # Getting scaler
    scaler_dir_path = config["global"]["SAVED_OBJECTS_PATH"]
    if scaler_dir_path[-1] not in ["\\", "/"]:
        scaler_dir_path += "/"
    scaler_file_path = scaler_dir_path + data_prefix + "_scaler.sav"
    scaler = pickle.load(open(scaler_file_path, "rb"))


    def scale(scaler, input_array, is_x=True):
        input_shape = input_array.shape
        if is_x:
            input_array = scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_shape)
        else:
            input_array = np.squeeze(input_array).flatten()
            transform = np.zeros((len(input_array), scaler.n_features_in_))
            transform[:,0] = input_array
            transform = scaler.transform(transform)
            input_array = transform[:,0].reshape(input_shape)
        return input_array

    train_x = scale(scaler, train_x)
    test_x = scale(scaler, test_x)
    if model_to_train!="classifier":
        train_y = scale(scaler, train_y, is_x=False)
        test_y = scale(scaler, test_y, is_x=False)

    def build_model(hp):
        model = tf.keras.Sequential()
        input_layer_name = '{}_input'.format(model_to_train)
        input_layer_nodes = hp.Int(input_layer_name, min_value=128, max_value=256, step=16)
        add_lstm_layers = hp.Int('num_lstm_layers', 0, 3)
        add_dense_layers = hp.Int('num_dense_layers', 0, 3)
        model.add(tf.keras.layers.LSTM(units=input_layer_nodes, 
                            return_sequences=True, input_shape=(train_x.shape[1:])))
        model.add(tf.keras.layers.Dropout(hp.Float('input_layer_dropout', min_value=0.0, max_value=0.5, step=0.1)))
        # Tune the number of hidden LSTM layers
        for i in range(add_lstm_layers):
            model.add(tf.keras.layers.LSTM(units=hp.Int('units_' + str(i), min_value=64, max_value=128, step=16), 
                              return_sequences=True))
            model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
        model.add(tf.keras.layers.Flatten())
        for i in range(add_dense_layers):
            model.add(tf.keras.layers.Dense(units=hp.Int('dense_units_' + str(i), min_value=16, max_value=64, step=8),
                            activation=hp.Choice('dense_activation_' + str(i), values=['relu', 'sigmoid'])))
             # Tune the inclusion of dropout layers
            model.add(tf.keras.layers.Dropout(hp.Float('dense_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
        if model_to_train == "classifier":
            model.add(tf.keras.layers.Dense(3, activation='softmax'))
            model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate_classifier', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        else:
            model.add(tf.keras.layers.Dense(train_y.shape[1]))
            model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                    loss='mse')
        return model
        
# Define the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=max_search,
    executions_per_trial=rerun_count,
    directory='model_tuning',
    project_name='{}_{}_regresssion_tuning'.format(data_prefix, model_to_train)
)

# Start the tuning process
tuner.search(train_x, train_y, epochs=70, validation_data=(test_x, test_y))