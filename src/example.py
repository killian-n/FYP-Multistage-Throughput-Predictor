import configparser
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
config = configparser.ConfigParser()
config.read('project.env')
module_path = config['global']['MODULE_PATH']
sys.path.append(module_path)

from trained_models.baseline import TrainedBaseline
from trained_models.multistage_all import MultistageAll
from trained_models.multistage_one import MultistageOne

if __name__ == "__main__":
    ###########
    # DATASETS
    ###########
    # All
    test_x = np.load("Datasets/Testing/act_all_network_test_x.npy")
    test_y = np.load("Datasets/Testing/act_all_network_test_y.npy")
    # Low
    low_test_x = np.load("Datasets/Testing/act_all_network_low_test_x.npy")
    low_test_y = np.load("Datasets/Testing/act_all_network_low_test_y.npy")
    # Medium
    medium_test_x = np.load("Datasets/Testing/act_all_network_medium_test_x.npy")
    medium_test_y = np.load("Datasets/Testing/act_all_network_medium_test_y.npy")
    # High
    high_test_x = np.load("Datasets/Testing/act_all_network_high_test_x.npy")
    high_test_y = np.load("Datasets/Testing/act_all_network_high_test_y.npy")
    #Classifier
    classifier_test_x = np.load("Datasets/Testing/act_all_network_classifier_test_x.npy")
    classifier_test_y = np.load("Datasets/Testing/act_all_network_classifier_test_y.npy")

    ###########
    # MODELS
    ###########
    baseline = TrainedBaseline("all_network_baseline")
    multiOne = MultistageOne("standardized_multiOne")
    multiAll = MultistageAll("standardized_multiAll")

    ###########
    # Baseline
    ###########
    baseline.set_scaler("src/saved.objects/all_network_features.sav")
    baseline.set_model(tf.keras.models.load_model("src/saved.objects/standard_4_all_network_baseline.hdf5"))
    baseline.set_test(test_x, test_y)
    baseline.set_low_test(low_test_x, low_test_y)
    baseline.set_medium_test(medium_test_x, medium_test_y)
    baseline.set_high_test(high_test_x, high_test_y)

    ###########
    # MultiOne
    ###########
    multiOne.set_classifier_scaler("src/saved.objects/all_network_features.sav")
    multiOne.set_low_scaler("src/saved.objects/solo_low_all_network_scaler.sav")
    multiOne.set_medium_scaler("src/saved.objects/solo_medium_all_network_scaler.sav")
    multiOne.set_high_scaler("src/saved.objects/solo_high_all_network_scaler.sav")
    multiOne.set_low_model("src/saved.objects/solo_low_all_network.hdf5")
    multiOne.set_medium_model("src/saved.objects/solo_medium_all_network.hdf5")
    multiOne.set_high_model("src/saved.objects/solo_high_all_network.hdf5")
    multiOne.set_classifier("src/saved.objects/solo_all_network_classifier.hdf5")
    multiOne.set_test(test_x, test_y)
    multiOne.set_low_test(low_test_x, low_test_y)
    multiOne.set_medium_test(medium_test_x, medium_test_y)
    multiOne.set_high_test(high_test_x, high_test_y)
    multiOne.set_classifier_test(classifier_test_x, classifier_test_y)

    ###########
    # MultiAll
    # ###########
    multiAll.set_classifier_scaler("src/saved.objects/all_network_features.sav")
    multiAll.set_low_scaler("src/saved.objects/solo_low_all_network_scaler.sav")
    multiAll.set_medium_scaler("src/saved.objects/solo_medium_all_network_scaler.sav")
    multiAll.set_high_scaler("src/saved.objects/solo_high_all_network_scaler.sav")
    multiAll.set_low_model("src/saved.objects/solo_low_all_network.hdf5")
    multiAll.set_medium_model("src/saved.objects/solo_medium_all_network.hdf5")
    multiAll.set_high_model("src/saved.objects/solo_high_all_network.hdf5")
    multiAll.set_classifier("src/saved.objects/solo_all_network_classifier.hdf5")
    multiAll.set_test(test_x, test_y)
    multiAll.set_low_test(low_test_x, low_test_y)
    multiAll.set_medium_test(medium_test_x, medium_test_y)
    multiAll.set_high_test(high_test_x, high_test_y)
    multiAll.set_classifier_test(classifier_test_x, classifier_test_y)


    ###########
    # Testing
    ###########
    baseline.test()
    multiOne.test()
    multiAll.test()