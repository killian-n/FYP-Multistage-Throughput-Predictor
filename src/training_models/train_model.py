import argparse
import configparser
import sys
import numpy as np
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)
from training_models.optimized_models import *
from training_models.standardized_models import *
from training_models.size_constraint_models import *


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train a model on pre-created datasets.')

    # Add the arguments to the argument parser
    parser.add_argument('--prefix', type=str, help='A file prefix for saved .hdf5 file of the selected model.')
    parser.add_argument('--model', type=str, help='Select which model to train. Values can be one of: [baseline, low, medium , high, classifier]')
    parser.add_argument('--data_prefix', type=str,
                         help="Optional argument. The file prefix of the saved train and test datasets found in Datasets directory. \n Assumed to be same as prefix if not provided")
    parser.add_argument('--use_balanced', type=bool, help="Optional argument. Specifies if baseline or classifier model should be trained on a balanced dataset. Default is False.")
    parser.add_argument('--standardise_models', type=bool, help="Optional argument. Specifies whether all models should use the same parameter design. Default is True")
    parser.add_argument('--size_constraint', type=int, help="Optional argument. Current options include [3, 1.5]")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    model_prefix = args.prefix
    model_to_train = args.model
    data_prefix = args.data_prefix
    use_balanced = args.use_balanced
    standardise = args.standardise_models
    size_constraint = args.size_constraint

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

    if not size_constraint:
        size_constraint=0

    if size_constraint not in [0,3,1.5]:
        size_constraint=0

    if standardise is None:
        standardise = True

    if standardise and size_constraint:
        print("overriding standardise with size_constraint")
        standardise = False

    model_name = model_prefix + "_" + model_to_train
    if model_to_train == "baseline":
        if standardise:
            model = StandarizedBaseline(model_name=model_name)
        elif size_constraint==3:
            model = Baseline3Mb(model_name=model_name)
        else:
            model = optimizedBaseline(model_name=model_name)
        saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_objects_path[-1] not in ["\\", "/"]:
            saved_objects_path += "/"
        model.set_scaler(saved_objects_path+"{}_scaler.sav".format(data_prefix))

    elif model_to_train == "low":
        if standardise:
            model = standardizedMultistageRegression(model_name=model_name)
        elif size_constraint==1.5:
            model = MultistageRegression1_5Mb(model_name=model_name)
        elif size_constraint==3:
            model = MultistageRegression3Mb(model_name=model_name)
        else:
            model = optimizedLowRegressionModel(model_name=model_name)
    
    elif model_to_train == "medium":
        if standardise:
            model = standardizedMultistageRegression(model_name=model_name)
        elif size_constraint==1.5:
            model = MultistageRegression1_5Mb(model_name=model_name)
        elif size_constraint==3:
            model = MultistageRegression3Mb(model_name=model_name)
        else:
            model = optimizedMediumRegressionModel(model_name=model_name)
    
    elif model_to_train == "high":
        if standardise:
            model = standardizedMultistageRegression(model_name=model_name)
        elif size_constraint==1.5:
            model = MultistageRegression1_5Mb(model_name=model_name)
        elif size_constraint==3:
            model = MultistageRegression3Mb(model_name=model_name)
        else:
            model = optimizedHighRegressionModel(model_name=model_name)
    
    else:
        saved_objects_path = config["global"]["SAVED_OBJECTS_PATH"]
        if saved_objects_path[-1] not in ["\\", "/"]:
            saved_objects_path += "/"
        if standardise:
            model = standardizedClassifierModel(model_name=model_name)
        elif size_constraint==1.5:
            model = ClassifierModel1_5Mb(model_name=model_name)
        elif size_constraint==3:
            model = ClassifierModel3Mb(model_name=model_name)
        else:
            model = optimizedClassifierModel(model_name=model_name)
        if use_balanced:
            model.set_class_weights(set_as_none=True)
        else:
            model.set_class_weights(filename="{}_class_weights.sav".format(data_prefix))
        model.set_scaler(saved_objects_path+"{}_scaler.sav".format(data_prefix))


    # Getting train and test Datasets
    train_dir_path = config["global"]["TRAINING_DATASETS_PATH"]
    if train_dir_path[-1] not in ["\\", "/"]:
        train_dir_path += "/"

    train_file_path = "{}{}_{}".format(train_dir_path, data_prefix, model_to_train)
    if model_to_train == "baseline":
        train_file_path = "{}{}".format(train_dir_path, data_prefix)
    if use_balanced:
        model.set_model_name(model_prefix+"_up"+model_to_train)
        train_x = np.load("{}_balanced_train_x.npy".format(train_file_path))
        train_y = np.load("{}_balanced_train_y.npy".format(train_file_path))
    else:
        train_x = np.load("{}_train_x.npy".format(train_file_path))
        train_y = np.load("{}_train_y.npy".format(train_file_path))

    model.set_train(train_x, train_y)
    model.build_model()
    model.train()