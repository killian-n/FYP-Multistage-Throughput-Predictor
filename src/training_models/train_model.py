import argparse
import configparser
import sys
import numpy as np
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from training_models.optimized_models import *


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train a model on pre-created datasets.')

    # Add the arguments to the argument parser
    parser.add_argument('--prefix', type=str, help='A file prefix for saved .hdf5 file of the selected model.')
    parser.add_argument('--model', type=str, help='Select which model to train. Values can be one of: [baseline, low, medium , high, classifier]')
    parser.add_argument('--data_prefix', type=str,
                         help="Optional argument. The file prefix of the saved train and test datasets found in Datasets directory. \n Assumed to be same as prefix if not provided")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    model_prefix = args.prefix
    model_to_train = args.model
    data_prefix = args.data_prefix

    if not model_prefix:
        print("Please provide a str prefix for the filename of the saved model and its datasets.")
        sys.exit(1)

    if not data_prefix:
        data_prefix = model_prefix
        print("Datasets prefix assumed to be the same as model prefix.")

    if not model_to_train:
        print("No model selected. Training baseline model.")

    if not model_to_train in ["baseline", "low", "medium", "high", "classifier"]:
        print("Selected model not in provided scope.")
        parser.print_help()
        sys.exit(1)


    model_name = model_prefix + "_" + model_to_train
    if model_to_train == "baseline":
        model = optimizedBaseline(model_name=model_name)
        saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_objects_path[-1] not in ["\\", "/"]:
            saved_objects_path += "/"
        model.set_scaler(saved_objects_path+"{}_scaler.sav".format(data_prefix))

    elif model_to_train == "low":
        model = optimizedLowRegressionModel(model_name=model_name)
    
    elif model_to_train == "medium":
        model = optimizedMediumRegressionModel(model_name=model_name)
    
    elif model_to_train == "high":
        model = optimizedHighRegressionModel(model_name=model_name)
    
    else:
        model = optimizedClassifierModel(model_name=model_name)
        model.set_class_weights(filename="{}_class_weights.sav")


    # Getting train and test Datasets
    train_dir_path = config["global"]["TRAINING_DATASETS_PATH"]
    if train_dir_path[-1] not in ["\\", "/"]:
        train_dir_path += "/"

    train_file_path = "{}{}_{}".format(train_dir_path, data_prefix, model_to_train)
    if model_to_train == "baseline":
        train_file_path = "{}{}".format(train_dir_path, data_prefix)
    train_x = np.load("{}_train_x.npy".format(train_file_path))
    train_y = np.load("{}_train_y.npy".format(train_file_path))
    model.set_train(train_x, train_y)
    model.build_model()
    model.train()