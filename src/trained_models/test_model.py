import argparse
import configparser
import sys
import numpy as np
import tensorflow as tf
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.baseline import TrainedBaseline
from trained_models.multistage_all import MultistageAll
from trained_models.multistage_one import MultistageOne

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train a model on pre-created datasets.')

    # Add the arguments to the argument parser
    parser.add_argument('--prefix', type=str, help='A file prefix for saved .hdf5 file of the selected model.')
    parser.add_argument('--model', type=str, help='Select which model to train. Values can be one of: [multiOne, mutliAll, baseline]')
    parser.add_argument('--data_prefix', type=str,
                         help="Optional argument. The file prefix of the saved train and test datasets found in Datasets directory. \n Assumed to be same as prefix if not provided")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    model_prefix = args.prefix
    model_to_test = args.model
    data_prefix = args.data_prefix

    if not model_prefix:
        print("Please provide a str prefix for the filename of the saved model and its datasets.")
        sys.exit(1)

    if not data_prefix:
        data_prefix = model_prefix
        print("Datasets prefix assumed to be the same as model prefix.")

    if not model_to_test:
        print("No model selected. Training baseline model.")

    if not model_to_test in ["multiOne", "multiAll", "baseline"]:
        print("Selected model not in provided scope.")
        parser.print_help()
        sys.exit(1)


    ###########
    # DATASETS
    ###########
    test_dir_path = config["global"]["TESTING_DATASETS_PATH"]
    if test_dir_path[-1] not in ["\\", "/"]:
        test_dir_path += "/"

    saved_objects_dir_path = config["global"]["SAVED_OBJECTS_PATH"]
    if saved_objects_dir_path[-1] not in ["\\", "/"]:
        saved_objects_dir_path += "/"
    # All
    test_x = np.load("{}{}_test_x.npy".format(test_dir_path, data_prefix))
    test_y = np.load("{}{}_test_y.npy".format(test_dir_path, data_prefix))
    # Low
    low_test_x = np.load("{}{}_low_test_x.npy".format(test_dir_path, data_prefix))
    low_test_y = np.load("{}{}_low_test_y.npy".format(test_dir_path, data_prefix))
    # Medium
    medium_test_x = np.load("{}{}_medium_test_x.npy".format(test_dir_path, data_prefix))
    medium_test_y = np.load("{}{}_medium_test_y.npy".format(test_dir_path, data_prefix))
    # High
    high_test_x = np.load("{}{}_high_test_x.npy".format(test_dir_path, data_prefix))
    high_test_y = np.load("{}{}_high_test_y.npy".format(test_dir_path, data_prefix))

    if model_to_test == "baseline":
        model = TrainedBaseline("{}_baseline".format(model_prefix))
        model.set_scaler("{}{}_scaler.sav".format(saved_objects_dir_path, data_prefix))
        model.set_model(tf.keras.models.load_model("{}{}_baseline.hdf5".format(saved_objects_dir_path, model_prefix)))
    
    else:
        #Classifier
        classifier_test_x = np.load("{}{}_classifier_test_x.npy".format(test_dir_path, data_prefix))
        classifier_test_y = np.load("{}{}_classifier_test_y.npy".format(test_dir_path, data_prefix))

        if model_to_test == "multiOne":
            model = MultistageOne("{}_multiOne".format(model_prefix))
        else: 
            model = MultistageAll("{}_multiAll".format(model_prefix))

        model.set_classifier_scaler("{}{}_scaler.sav".format(saved_objects_dir_path, data_prefix))
        model.set_low_scaler("{}{}_low_scaler.sav".format(saved_objects_dir_path, data_prefix))
        model.set_medium_scaler("{}{}_medium_scaler.sav".format(saved_objects_dir_path, data_prefix))
        model.set_high_scaler("{}{}_high_scaler.sav".format(saved_objects_dir_path, data_prefix))
        model.set_low_model("{}{}_low.hdf5".format(saved_objects_dir_path, data_prefix))
        model.set_medium_model("{}{}_medium.hdf5".format(saved_objects_dir_path, data_prefix))
        model.set_high_model("{}{}_high.hdf5".format(saved_objects_dir_path, data_prefix))
        model.set_classifier("{}{}_classifier.hdf5".format(saved_objects_dir_path, data_prefix))
        model.set_classifier_test(classifier_test_x, classifier_test_y)
        

    model.set_test(test_x, test_y)
    model.set_low_test(low_test_x, low_test_y)
    model.set_medium_test(medium_test_x, medium_test_y)
    model.set_high_test(high_test_x, high_test_y)

    model.test()