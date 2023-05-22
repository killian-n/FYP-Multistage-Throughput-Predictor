import argparse
import configparser
import sys
import numpy as np
import tensorflow as tf
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
    else:
        train_x = np.load("{}_train_x.npy".format(train_file_path))
        train_y = np.load("{}_train_y.npy".format(train_file_path))

    test_dir_path = config["global"]["TESTING_DATASETS_PATH"]
    if test_dir_path[-1] not in ["\\", "/"]:
        test_dir_path += "/"

    test_file_path = "{}{}_{}".format(test_dir_path, data_prefix, model_to_train)
    if model_to_train == "baseline":
        test_file_path = "{}{}".format(test_dir_path, data_prefix)
    else:
        test_x = np.load("{}_test_x.npy".format(test_file_path))
        test_y = np.load("{}_test_y.npy".format(test_file_path))

    def build_model(hp):
        input_layer_nodes = hp.Int('{}_input'.format(model_to_train), min_value=128, max_value=256, step=16)
        add_lstm_layers = hp.Int('num_dense_layers', 0, 3)
        add_dense_layers = hp.Int('num_dense_layers', 0, 3)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=input_layer_nodes, 
                            return_sequences=True, input_shape=(train_x.shape[1:])))
        model.add(tf.keras.layers.Dropout(hp.Float('input_layer_dropout', min_value=0.0, max_value=0.5, step=0.1)))
        # Tune the number of hidden LSTM layers
        for i in range(add_lstm_layers):
            add_lstm_nodes = hp.Int('units_' + str(i), min_value=64, max_value=int(input_layer_nodes), step=8)
            model.add(tf.keras.layers.LSTM(units=add_lstm_nodes, 
                                return_sequences=True))
            model.add(tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))
        model.add(tf.keras.layers.Flatten())
        # Tuning for addition dense layers
        for i in range(add_dense_layers):
            if add_lstm_layers:
                add_dense_nodes = hp.Int("dense_units_"+str(i),min_value=16, max_value=int(add_lstm_nodes))
            else:
                add_dense_nodes = hp.Int("dense_units_"+str(i),min_value=16, max_value=int(input_layer_nodes))
            model.add(tf.keras.layers.Dense(units=add_dense_nodes,
                                activation=hp.Choice('dense_activation_' + str(i), values=['relu', 'sigmoid'])))
            # Tune the inclusion of dropout layers
            model.add(tf.keras.layers.Dropout(hp.Float('dense_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

        if model_to_train == "classifier":
            model.add(tf.keras.layers.Dense(3, activation='softmax'))
            model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
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
    project_name='{}_regresssion_tuning'.format(model_to_train)
)

# Start the tuning process
tuner.search(train_x, train_y, epochs=70, validation_data=(test_x, test_y))